'''Test MNIST model'''
from __future__ import print_function

import os
import time

import foolbox as fb
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import yaml

from adv.dataset_utils import load_mnist_all
from adv.mnist_model import BasicModel
from adv.pgd_attack import PGDAttack, pgdp
from adv.utils import classify, get_acc, get_logger, quantize


def bb_attack(netfb, x_test, y_test, x_init, config):
    attack = fb.attacks.LinfinityBrendelBethgeAttack(
        init_attack=None, steps=1000, lr=0.001, binary_search_steps=5)
    num_batches = int(np.ceil(x_test.size(0) / config['test']['batch_size']))
    x_adv = x_test.clone()
    for i in range(num_batches):
        begin = i * config['test']['batch_size']
        end = (i + 1) * config['test']['batch_size']
        criterion = fb.criteria.Misclassification(y_test[begin:end])
        x_adv[begin:end] = attack(
            netfb, x_test[begin:end], criterion,
            epsilons=config['pgd']['epsilon'],
            starting_points=x_init[begin:end])[1]
    return x_adv


def pgd_init(netfb, x_test, y_test, device, batch_size):

    num_batches = int(np.ceil(x_test.size(0) / batch_size))
    x_init = x_test.clone()
    for i in range(num_batches):
        begin, end = i * batch_size, (i + 1) * batch_size
        pgd_init_batch(netfb, x_test[begin:end], y_test[begin:end],
                       x_init[begin:end], device)
    return x_init


def pgd_init_batch(netfb, x_test, y_test, x_init, device):

    total = x_test.size(0)
    init_attack = fb.attacks.PGD(
        rel_stepsize=0.033333, steps=100, random_start=True)
    idx_adv = np.zeros(total)
    i = 1

    while idx_adv.sum() < total:
        eps = i * 0.1
        if eps >= 1:
            break
        idx = np.where(1 - idx_adv)[0]
        init_ctr = fb.criteria.Misclassification(y_test[idx].cuda())
        x_pgd = init_attack(
            netfb, x_test[idx].to(device), init_ctr, epsilons=eps)
        for j, jj in enumerate(idx):
            x_init[jj] = x_pgd[0][j].cpu()
            idx_adv[jj] = x_pgd[2][j]
        i += 1

    y_adv = netfb(x_init.to(device)).argmax(1)
    # in case not all samples are adversarial
    idx_not_adv = np.where(1 - idx_adv)[0]
    for idx in idx_not_adv:
        indices = np.where((y_adv[idx] != y_adv).cpu()
                           & idx_adv.astype(np.bool))[0]
        i = np.random.choice(indices)
        x_init[idx] = x_init[i]


def main():
    """Main function. Use config file test_mnist.yml"""

    # Parse config file
    with open('test_mnist.yml', 'r') as stream:
        config = yaml.safe_load(stream)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = config['meta']['gpu_id']

    # Set experiment id
    exp_id = config['meta']['exp_id']
    model_name = config['meta']['model_name'] + str(exp_id)

    # Set all random seeds
    seed = config['meta']['seed']
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Set up model directory
    save_dir = os.path.join(config['meta']['save_path'], 'saved_models')
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, model_name)

    log = get_logger('test_' + model_name, 'test_cifar10')
    log.info('\n%s', yaml.dump(config))
    log.info('Preparing data...')
    (x_train, y_train), (_, _), (x_test, y_test) = load_mnist_all(
        data_dir=config['meta']['data_path'], val_size=0.1, shuffle=False,
        seed=seed)
    num_classes = 10

    log.info('Building model...')
    net = BasicModel()
    net = net.eval().to(device)

    # have to handle model loaded from CAT18 differently
    if 'cat18' in model_path:
        net = torch.nn.DataParallel(net)
        net.load_state_dict(torch.load(model_path + '.pt')['net'])
        net = net.module
    else:
        net.load_state_dict(torch.load(model_path + '.pt'))

    if device == 'cuda':
        if len(config['meta']['gpu_id']) > 1:
            net = torch.nn.DataParallel(net)
            net = net.eval()
        cudnn.benchmark = True

    num_test_samples = config['test']['num_test_samples']
    if config['pgd']['quant']:
        y_pred = classify(net, quantize(
            x_test[:num_test_samples]), num_classes=num_classes)
    else:
        y_pred = classify(
            net, x_test[:num_test_samples], num_classes=num_classes)
    acc = get_acc(y_pred, y_test[:num_test_samples])
    start_time = time.time()
    if config['bb']:
        log.info('Starting Brendel & Bethge attack...')
        netfb = fb.PyTorchModel(net, bounds=(0, 1), preprocessing=dict())
        x_init = pgd_init(
            netfb, x_test[:num_test_samples], y_test[:num_test_samples],
            device, config['test']['batch_size'])
        log.info('Finish an initial attack with PGD. Running BB attack...')
        x_adv = bb_attack(
            netfb, x_test[:num_test_samples].to(device),
            y_test[:num_test_samples].to(device), x_init.to(device), config)
        if config['pgd']['quant']:
            x_adv = quantize(x_adv)
        y_pred = classify(net, x_adv, num_classes=num_classes)
        adv_acc = get_acc(y_pred, y_test[:num_test_samples])
    elif config['pgd']['plus']:
        log.info('Starting PGD+ attack...')
        adv_acc = pgdp(net, x_train, y_train, x_test[:num_test_samples],
                       y_test[:num_test_samples], config['test']['batch_size'],
                       config['pgd'], num_classes=num_classes)
    else:
        log.info('Starting PGD attack...')
        attack = PGDAttack(net, x_train, y_train)
        x_adv = attack(x_test[:num_test_samples], y_test[:num_test_samples],
                       batch_size=config['test']['batch_size'],
                       **config['pgd'])
        y_pred = classify(net, x_adv, num_classes=num_classes)
        adv_acc = get_acc(y_pred, y_test[:num_test_samples])
    log.info('Clean acc: %.4f, adv acc: %.4f.', acc, adv_acc)
    log.info('Attack runtime: %.4fs', time.time() - start_time)


if __name__ == '__main__':
    main()
