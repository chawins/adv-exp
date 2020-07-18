'''Train MNIST model with adversarial training'''
from __future__ import print_function

import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml

from adv.adv_model import FGSMModel, PGDModel
from adv.dataset_utils import load_mnist
from adv.mnist_model import BasicModel, BatchNormModel
from adv.utils import get_logger, trades_loss


def evaluate(net, dataloader, criterion, device, adv=False):
    """Evaluate network."""

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
    """Main training function."""

    net.train()
    train_loss = 0
    train_correct = 0
    train_total = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        # Pass training samples to the adversarial model
        outputs = net(inputs, targets, params=config['at'])
        if config['at']['loss_func'] == 'trades':
            # Compute TRADES loss
            outputs_clean = net.basic_net(inputs)
            loss = trades_loss((outputs_clean, outputs), targets, config['at'])
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
    adv_loss, adv_acc = evaluate(net, validloader, criterion, device, adv=True)
    val_loss, val_acc = evaluate(
        net, validloader, criterion, device, adv=False)

    log.info(' %5d | %.4f, %.4f | %.4f, %.4f | %.4f, %.4f | ', epoch,
             train_loss / train_total, train_correct / train_total,
             adv_loss, adv_acc, val_loss, val_acc)

    # Save model weights
    if not config['train']['save_best_only']:
        # Save model every <save_epochs> epochs
        if epoch % config['train']['save_epochs'] == 0:
            log.info('Saving model...')
            torch.save(net.module.basic_net.state_dict(),
                       model_path + '_epoch%d.pt' % epoch)
    elif config['train']['save_best_only'] and adv_acc > best_acc:
        # Save only the model with the highest adversarial accuracy
        log.info('Saving model...')
        torch.save(net.module.basic_net.state_dict(), model_path + '.pt')
        best_acc = adv_acc
    return best_acc


def main(config=None):
    """Main function. Use config file train_mnist.yml"""

    # Parse config file
    if not config:
        with open('train_mnist.yml', 'r') as stream:
            config = yaml.safe_load(stream)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = config['meta']['gpu_id']

    # Set experiment id
    exp_id = config['meta']['exp_id']
    model_name = config['meta']['model_name'] + str(exp_id)

    # Training parameters
    epochs = config['train']['epochs']
    lr = config['train']['learning_rate']

    # AT params
    at_params = config['at']

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

    # Set up logger
    log = get_logger(model_name, 'train_mnist')
    log.info('\n%s', yaml.dump(config))

    # Load dataset
    log.info('Preparing data...')
    trainloader, validloader, testloader = load_mnist(
        config['train']['batch_size'],
        data_dir=config['meta']['data_path'],
        val_size=0.1, shuffle=True, seed=seed)

    # Build neural network
    log.info('Building model...')
    basic_net = BatchNormModel().to(device)

    # Wrap the neural network with module that generates adversarial examples
    if config['at']['method'] == 'pgd' or config['at']['method'] == 'none':
        net = PGDModel(basic_net, config['at'])
    elif config['at']['method'] == 'fgsm':
        net = FGSMModel(basic_net, config['at'])
    else:
        raise NotImplementedError('Specified AT method not implemented.')

    # If GPU is available, allows parallel computation and cudnn speed-up
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    # Specify loss function of the network
    criterion = nn.CrossEntropyLoss()

    # Set up optimizer
    optimizer = optim.SGD(
        basic_net.parameters(), lr=lr, momentum=0.9,
        weight_decay=config['train']['l2_reg'])

    # Set up learning rate schedule
    if config['train']['lr_scheduler'] == 'cyclic':
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, lr, epochs=epochs, steps_per_epoch=422,
            pct_start=0.5, anneal_strategy='linear', cycle_momentum=False,
            base_momentum=0.9, div_factor=1e5, final_div_factor=1e5)
    elif config['train']['lr_scheduler'] == 'step':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, [40, 50, 60], gamma=0.1)
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
  
    # Evaluate network on clean data
    test_loss, test_acc = evaluate(
        net, testloader, criterion, device, adv=False)
    log.info('Test loss: %.4f, Test acc: %.4f', test_loss, test_acc)

    # Evaluate network on adversarial data
    test_loss, test_acc = evaluate(
        net, testloader, criterion, device, adv=True)
    log.info('Test adv loss: %.4f, Test adv acc: %.4f', test_loss, test_acc)
    return net    

def ensemble():
    models = []
    log = get_logger("ensemble", 'train_mnist')

    config_modifications = [
        {
            'meta': {
                'model_name': 'mnist_exp_at_seed220',
                'seed': 220,
            }
            'at': {
                'method': 'pgd',
            }
        }, 
        {
            'meta': {
                'model_name': 'mnist_exp_at_seed2020',
                'seed': 2020,
            }
            'at': {
                'method': 'pgd',
            }
        }, 
        {
            'meta': {
                'model_name': 'mnist_exp_at_seed1000',
                'seed': 1000,
            }
            'at': {
                'method': 'pgd',
            }
        }, 
        {
            'meta': {
                'model_name': 'mnist_exp_at_seed1020_fgsm',
                'seed': 1020,
            }
            'at': {
                'method': 'fgsm',
            }
        }, 
        {
            'meta': {
                'model_name': 'mnist_exp_at_seed1010_none',
                'seed': 1010,
            }
            'at': {
                'method': 'none',
            }
        },
    ]

    for config_mod in config_modifications:
        with open("train_mnist.yml") as stream:
            config = yaml.safe_load(stream)
        
        # For each additional field you wish to modify, add a line here
        config['meta']['model_name'] = config_mod['meta']['model_name']
        config['meta']['seed'] = config_mod['meta']['seed']
        config['at']['method'] = config_mod['at']['method']
        log.info('name: %s, seed: %d, method: %s, epoch: %d', 
    	    config['meta']['model_name'], config['meta']['seed'], config['at']['method'], config['train']['epochs'])
        models.append(main(config))

    # with open("train_mnist.yml") as stream:
    #     config = yaml.safe_load(stream)

    # config['meta']['model_name'] = 'mnist_exp_at_seed220'
    # config['meta']['seed'] = 220
    # config['at']['method'] = 'pgd'
    # with open("train_mnist.yml", "w") as f:
    #     yaml.dump(config, f)
    # log.info('name: %s, seed: %d, method: %s, epoch: %d', 
    # 	config['meta']['model_name'], config['meta']['seed'], config['at']['method'], config['train']['epochs'])
    # models.append(main())

    # config['meta']['model_name'] = 'mnist_exp_at_seed2020'
    # config['meta']['seed'] = 2020
    # config['at']['method'] = 'pgd'
    # with open("train_mnist.yml", "w") as f:
    #     yaml.dump(config, f)
    # log.info('name: %s, seed: %d, method: %s, epoch: %d', 
    # 	config['meta']['model_name'], config['meta']['seed'], config['at']['method'], config['train']['epochs'])
    # models.append(main())

    # config['meta']['model_name'] = 'mnist_exp_at_seed1000_epoch7'
    # config['meta']['seed'] = 1000
    # config['at']['method'] = 'pgd'
    # config['train']['epochs'] = 7
    # with open("train_mnist.yml", "w") as f:
    #     yaml.dump(config, f)
    # log.info('name: %s, seed: %d, method: %s, epoch: %d', 
    # 	config['meta']['model_name'], config['meta']['seed'], config['at']['method'], config['train']['epochs'])
    # models.append(main())

    # config['meta']['model_name'] = 'mnist_exp_at_seed1020'
    # config['meta']['seed'] = 1020
    # config['at']['method'] = 'pgd'
    # with open("train_mnist.yml", "w") as f:
    #     yaml.dump(config, f)
    # log.info('name: %s, seed: %d, method: %s, epoch: %d', 
    # 	config['meta']['model_name'], config['meta']['seed'], config['at']['method'], config['train']['epochs'])
    # models.append(main())

    # config['meta']['model_name'] = 'mnist_exp_at_seed1010_none'
    # config['meta']['seed'] = 1010
    # config['at']['method'] = 'none'
    # config['train']['epochs'] = 7
    # with open("train_mnist.yml", "w") as f:
    #     yaml.dump(config, f)
    # log.info('name: %s, seed: %d, method: %s, epoch: %d', 
    # 	config['meta']['model_name'], config['meta']['seed'], config['at']['method'], config['train']['epochs'])
    # models.append(main())

    trainloader, validloader, testloader = load_mnist(
        config['train']['batch_size'],
        data_dir=config['meta']['data_path'],
        val_size=0.1, shuffle=True, seed=config['meta']['seed'])
    return models, testloader

def evaluate_ensemble(models, testloader, adv):
    criterion = nn.CrossEntropyLoss()
    device = 'cuda'
    net.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    predictions = {}
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            cur_output = []
            for net in models:
            	outputs = net(inputs, targets, adv=adv)
            	cur_output.append(outputs)
            predictions[batch_idx] = np.round(np.mean(cur_output, axis=0))
            loss = criterion(predictions[batch_idx], targets)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += 1
            val_correct += predicted.eq(targets).float().mean().item()
    return predictions, val_loss / val_total, val_correct / val_total


if __name__ == '__main__':
    models, testloader = ensemble()
    predictions, test_loss, test_acc = evaluate_ensemble(models, testloader, True)
    log.info('Test loss: %.4f, Test acc: %.4f', test_loss, test_acc)
    
