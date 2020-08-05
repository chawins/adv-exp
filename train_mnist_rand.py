'''Train MNIST model with adversarial training'''
from __future__ import print_function

import argparse
import os
import pprint

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml

from adv.adv_model import FGSMModel, PGDModel
from adv.dataset_utils import load_mnist
from adv.mnist_model import BasicModel, BatchNormModel
from adv.random_model import RandModel
from adv.utils import get_logger, trades_loss


def evaluate(net, dataloader, criterion, config, device, clean=True):
    """Evaluate network."""

    net.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            if config['meta']['method'] == 'none':
                outputs = net(inputs, targets, adv=False)
                loss = criterion(outputs, targets)
            elif config['meta']['method'] == 'rand':
                outputs = net(inputs, rand=(not clean))
                if (config['rand']['rule'] in ['mean_probs', 'mean_logits']
                        and not clean):
                    loss = criterion(outputs.log(), targets)
                else:
                    loss = F.cross_entropy(outputs, targets)
            elif config['meta']['method'] in ['pgd', 'at', 'pgd-rand']:
                outputs = net(inputs, targets, adv=(not clean))
                if (config['rand']['rule'] in ['mean_probs', 'mean_logits']
                        and config['meta']['method'] == 'pgd-rand'):
                    loss = criterion(outputs.log(), targets)
                else:
                    loss = F.cross_entropy(outputs, targets)
            else:
                outputs = net(inputs)
                loss = criterion(outputs, targets)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += 1
            val_correct += predicted.eq(targets).float().mean().item()

    return val_loss / val_total, val_correct / val_total


def train(net, trainloader, validloader, criterion, optimizer, config,
          epoch, device, log, best_acc, model_path, lr_scheduler=None):
    """Main training function."""

    net.train()
    logsoftmax = False
    train_loss = 0
    train_correct = 0
    train_total = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        # Pass training samples to the adversarial model
        if config['meta']['method'] == 'none':
            outputs = net(inputs, targets, adv=False)
        elif config['meta']['method'] == 'rand':
            outputs = net(inputs, rand=True)
            if config['rand']['rule'] in ['mean_probs', 'mean_logits']:
                outputs = outputs.log()
        elif config['meta']['method'] in ['pgd', 'fgsm', 'pgd-rand']:
            outputs = net(inputs, targets, adv=True)
            if (config['rand']['rule'] in ['mean_probs', 'mean_logits']
                    and config['meta']['method'] == 'pgd-rand'):
                outputs = outputs.log()
                logsoftmax = True
        else:
            raise NotImplementedError('Specified method not implemented.')

        if config['at']['loss_func'] == 'trades':
            # Compute TRADES loss
            logits_clean = net.basic_net(inputs)
            loss = trades_loss(
                (logits_clean, outputs), targets, config['at'], logsoftmax)
        else:
            # Use cross entropy loss
            loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        train_total += 1
        train_correct += predicted.eq(targets).float().mean().item()

    # Compute loss and accuracy on validation set
    adv_loss, adv_acc = evaluate(
        net, validloader, criterion, config, device, clean=False)
    val_loss, val_acc = evaluate(
        net, validloader, criterion, config, device, clean=True)

    log.info(train_correct, train_total)

    log.info(' %5d | %.4f, %.4f | %.4f, %.4f | %.4f, %.4f | ', epoch,
             train_loss / train_total, train_correct / train_total,
             adv_loss, adv_acc, val_loss, val_acc)

    # Save model weights
    if config['meta']['method'] == 'pgd-rand':
        state_dict = net.module.basic_net.basic_net.state_dict()
    else:
        state_dict = net.module.basic_net.state_dict()
    if not config['meta']['save_best_only']:
        # Save model every <save_epochs> epochs
        if epoch % config['meta']['save_epochs'] == 0:
            log.info('Saving model...')
            torch.save(state_dict, model_path + '_epoch%d.pt' % epoch)
    elif config['meta']['save_best_only'] and adv_acc > best_acc:
        # Save only the model with the highest adversarial accuracy
        log.info('Saving model...')
        torch.save(state_dict, model_path + '.pt')
        best_acc = adv_acc
    return best_acc


def main(config_file):
    """Main function. Use config file train_mnist.yml"""

    # Parse config file
    with open(config_file, 'r') as stream:
        config = yaml.safe_load(stream)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = config['meta']['gpu_id']

    # Set experiment id
    exp_id = config['meta']['exp_id']
    model_name = config['meta']['model_name'] + str(exp_id)

    # Training parameters
    epochs = config['meta']['epochs']
    lr = config['meta']['learning_rate']
    if config['meta']['method'] in ['rand', 'pgd-rand']:
        # Copy normalization to RandModel's params
        config['rand']['normalize'] = config['meta']['normalize']
        # Normalization is done after transformation only
        config['meta']['normalize'] = None
        config['at']['normalize'] = None
    elif config['meta']['method'] == 'pgd':
        # Copy normalization to PGDModel's params
        config['at']['normalize'] = config['meta']['normalize']
        # Normalization is done after perturbation only
        config['meta']['normalize'] = None
    else:
        config['rand']['normalize'] = None
        config['at']['normalize'] = None

    # AT params
    at_params = config['at']

    # Set all random seeds
    seed = config['meta']['seed']
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Set up model directory
    save_dir = os.path.join(config['meta']['save_path'], 'rand_models')
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, model_name)

    # Set up logger
    log = get_logger(model_name, 'train_mnist')
    log.info('\n%s', pprint.pformat(config))
    # log.info('\n%s', yaml.dump(config))

    # Load dataset
    log.info('Preparing data...')
    # Differs from Chawin's latest version (no `load_dataset`)
    # (trainloader, validloader, testloader), num_classes = load_dataset(
    #     config['meta'], 'train')
    trainloader, validloader, testloader = load_mnist(
        config['meta']['batch_size'],
        data_dir=config['meta']['data_path'],
        val_size=0.1, shuffle=True, seed=seed)

    # Build neural network
    # Differs from Chawin's latest version (no `create_model`)
    log.info('Building model...')
    rand_params = config['rand']
    basic_net = RandModel(BatchNormModel().to(device), rand_params).to(device)

    # Wrap the neural network with module that generates adversarial examples
    # Differs from Chawin's latest version (no `create_wrapper`)
    if config['at']['method'] == 'pgd' or config['at']['method'] == 'none':
        net = PGDModel(basic_net, config['at'])
    elif config['at']['method'] == 'fgsm':
        net = FGSMModel(basic_net, config['at'])
    else:
        raise NotImplementedError('Specified AT method not implemented.')

    print(device)

    # If GPU is available, allows parallel computation and cudnn speed-up
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    # Specify loss function of the network
    if (config['rand']['rule'] in ['mean_probs', 'mean_logits']
            and config['meta']['method'] in ['rand', 'pgd-rand']):
        criterion = nn.NLLLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    # Set up optimizer
    optimizer = optim.SGD(
        basic_net.parameters(), lr=lr, momentum=0.9,
        weight_decay=config['meta']['l2_reg'])

    # Set up learning rate schedule
    if config['meta']['lr_scheduler'] == 'cyclic':
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, lr, epochs=epochs, steps_per_epoch=422,
            pct_start=0.5, anneal_strategy='linear', cycle_momentum=False,
            base_momentum=0.9, div_factor=1e5, final_div_factor=1e5)
    elif config['meta']['lr_scheduler'] == 'step':
        if epochs <= 70:
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, [40, 50, 60], gamma=0.1)
        elif epochs <= 100:
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, [40, 60, 80], gamma=0.1)
        elif epochs <= 160:
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, [60, 80, 100, 120, 140], gamma=0.2)
        else:
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, [60, 120, 160], gamma=0.2)
    else:
        lr_scheduler = None

    # Starting the main training loop over epochs
    log.info(' epoch | loss  , acc    | adv_l , adv_a  | val_l , val_a  |')
    best_acc = 0
    config['at']['gap'] = config['at']['init_gap']
    for epoch in range(epochs):

        if config['at']['step_gap'] is not None:
            # Increase probability gap in steps when using ATES
            if epoch in config['at']['step_gap']:
                config['at']['gap'] += (config['at']['final_gap']
                                        - config['at']['init_gap']) / len(config['at']['step_gap'])
        elif config['at']['linear_gap'] is not None:
            # Increase probability gap linearly when using ATES
            lin_gap = config['at']['linear_gap']
            interval = lin_gap[1] - lin_gap[0]
            if lin_gap[0] <= epoch < lin_gap[1]:
                config['at']['gap'] += (config['at']['final_gap']
                                        - config['at']['init_gap']) / interval

        # calculate fosc threshold if Dynamic AT is used
        if config['at']['use_fosc']:
            fosc_thres = config['at']['fosc_max'] * \
                (1 - (epoch / config['at']['dynamic_epoch']))
            config['at']['fosc_thres'] = np.maximum(0, fosc_thres)

        if config['meta']['lr_scheduler'] == 'step':
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

    # Evaluate network on clean data
    test_loss, test_acc = evaluate(
        net, testloader, criterion, config, device, clean=True)
    log.info('Test loss: %.4f, Test acc: %.4f', test_loss, test_acc)

    # Evaluate network on adversarial data
    test_loss, test_acc = evaluate(
        net, testloader, criterion, device, clean=False)
    log.info('Test adv loss: %.4f, Test adv acc: %.4f', test_loss, test_acc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test robustness')
    parser.add_argument(
        'config_file', type=str, help='name of config file')
    args = parser.parse_args()
    main(args.config_file)
