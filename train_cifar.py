'''Train CIFAR-10 model with adversarial training'''
from __future__ import print_function

import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml

from lib.adv_model import FGSMModel, PGDModel
from lib.cifar10_model import PreActBlock, PreActResNet
from lib.dataset_utils import load_cifar10, load_cifar100
from lib.utils import get_logger
from lib.wideresnet import WideResNet


def trades_loss(logits, targets, params):
    loss_natural = F.cross_entropy(logits[0], targets)
    loss_robust = F.kl_div(F.log_softmax(logits[1], dim=1),
                           F.softmax(logits[0], dim=1))
    return loss_natural + params['beta'] * loss_robust


def evaluate(net, dataloader, criterion, device, config, adv=False):

    net.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs, targets, adv=adv)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += 1
            val_correct += predicted.eq(targets).float().mean().item()

    return val_loss / val_total, val_correct / val_total


def train(net, trainloader, validloader, criterion, optimizer, config,
          epoch, device, log, best_acc, model_path, lr_scheduler=None):

    net.train()
    train_loss = 0
    train_correct = 0
    train_total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs, targets, params=config['at'])
        if config['at']['loss_func'] == 'trades':
            outputs_clean = net.basic_net(inputs)
            loss = trades_loss((outputs_clean, outputs), targets, config['at'])
        else:
            loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # DEBUG
        # print(loss)
        # print(outputs)

        if lr_scheduler is not None:
            lr_scheduler.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        train_total += 1
        train_correct += predicted.eq(targets).float().mean().item()

    adv_loss, adv_acc = evaluate(
        net, validloader, criterion, device, config, adv=True)
    val_loss, val_acc = evaluate(
        net, validloader, criterion, device, config, adv=False)

    log.info(' %5d | %.4f, %.4f | %.4f, %.4f | %.4f, %.4f | ', epoch,
             train_loss / train_total, train_correct / train_total,
             adv_loss, adv_acc, val_loss, val_acc)

    # Save model weights
    if not config['train']['save_best_only']:
        if epoch % config['train']['save_epochs'] == 0:
            log.info('Saving model...')
            torch.save(net.basic_net.module.state_dict(),
                       model_path + '_epoch%d.pt' % epoch)
    elif config['train']['save_best_only'] and adv_acc > best_acc:
        log.info('Saving model...')
        torch.save(net.basic_net.module.state_dict(), model_path + '.pt')
        best_acc = adv_acc
    return best_acc


def main():
    """Main function. Use config file train_cifar.yml"""

    # Parse config file
    with open('train_cifar.yml', 'r') as stream:
        config = yaml.safe_load(stream)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = config['meta']['gpu_id']

    # Set experiment id
    exp_id = config['meta']['exp_id']
    model_name = config['meta']['model_name'] + str(exp_id)

    # Training parameters
    epochs = config['train']['epochs']
    lr = config['train']['learning_rate']

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

    log = get_logger(model_name, 'train_cifar10')
    log.info('\n%s', yaml.dump(config))
    log.info('Preparing data...')
    if config['train']['dataset'] == 'cifar10':
        trainloader, validloader, testloader = load_cifar10(
            config['train']['batch_size'], data_dir=config['meta']['data_path'],
            val_size=0.1, normalize=False, augment=config['train']['data_aug'],
            shuffle=False, seed=seed)
        num_classes = 10
    elif config['train']['dataset'] == 'cifar100':
        trainloader, validloader, testloader = load_cifar100(
            config['train']['batch_size'], data_dir=config['meta']['data_path'],
            val_size=0.1, augment=config['train']['data_aug'], shuffle=False,
            seed=seed)
        num_classes = 100
    else:
        raise NotImplementedError('invalid dataset.')

    log.info('Building model...')
    if config['train']['network'] == 'resnet':
        # use ResNetV2-20
        basic_net = PreActResNet(
            PreActBlock, [2, 2, 2, 2], num_classes=num_classes)
    elif config['train']['network'] == 'wideresnet':
        # use WideResNet-28-10
        basic_net = WideResNet(num_classes=num_classes)
    else:
        raise NotImplementedError('Specified network not implemented.')

    basic_net = basic_net.to(device)
    if device == 'cuda':
        basic_net = torch.nn.DataParallel(basic_net)
        cudnn.benchmark = True

    if config['at']['method'] == 'pgd' or config['at']['method'] == 'none':
        net = PGDModel(basic_net, config['at'])
    elif config['at']['method'] == 'fgsm':
        net = FGSMModel(basic_net, config['at'])
    else:
        raise NotImplementedError('Specified AT method not implemented.')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        basic_net.module.parameters(), lr=lr, momentum=0.9,
        weight_decay=config['train']['l2_reg'])
    if config['train']['lr_scheduler'] == 'cyclic':
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, lr, epochs=epochs, steps_per_epoch=352,
            pct_start=config['train']['pct_start'], anneal_strategy='linear',
            cycle_momentum=False, base_momentum=0.9, div_factor=1e5,
            final_div_factor=1e5)
    elif config['train']['lr_scheduler'] == 'step':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, [40, 60, 80], gamma=0.1)
    else:
        lr_scheduler = None

    log.info(' epoch | loss  , acc    | adv_l , adv_a  | val_l , val_a  |')
    best_acc = 0
    config['at']['gap'] = config['at']['init_gap']
    for epoch in range(epochs):
        # if specified, increase "gap" in steps when using ATES
        if config['at']['step_gap'] is not None:
            if epoch in config['at']['step_gap']:
                config['at']['gap'] += (config['at']['final_gap']
                                        - config['at']['init_gap']) / len(config['at']['step_gap'])
        elif config['at']['linear_gap'] is not None:
            lin_gap = config['at']['linear_gap']
            interval = lin_gap[1] - lin_gap[0]
            if lin_gap[0] <= epoch < lin_gap[1]:
                config['at']['gap'] += (config['at']['final_gap']
                                        - config['at']['init_gap']) / interval

        # calculate fosc threshold if Dynamic AT is used
        if config['at']['use_fosc']:
            fosc_thres = config['at']['fosc_max'] * \
                (1 - epoch / config['at']['dynamic_epoch'])
            config['at']['fosc_thres'] = np.maximum(0, fosc_thres)

        if config['train']['lr_scheduler'] == 'step':
            best_acc = train(
                net, trainloader, validloader, criterion, optimizer, config,
                epoch, device, log, best_acc, model_path,
                lr_scheduler=None)
            lr_scheduler.step()
        else:
            best_acc = train(
                net, trainloader, validloader, criterion, optimizer, config,
                epoch, device, log, best_acc, model_path,
                lr_scheduler=lr_scheduler)

    test_loss, test_acc = evaluate(
        net, testloader, criterion, device, config, adv=False)
    log.info('Test loss: %.4f, Test acc: %.4f', test_loss, test_acc)
    test_loss, test_acc = evaluate(
        net, testloader, criterion, device, config, adv=True)
    log.info('Test adv loss: %.4f, Test adv acc: %.4f', test_loss, test_acc)


if __name__ == '__main__':
    main()
