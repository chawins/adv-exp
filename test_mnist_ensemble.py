from __future__ import print_function

import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml

from adv.dataset_utils import load_mnist_all
from adv.random_model import RandModel
from adv.mnist_model import BasicModel, BatchNormModel, EnsembleModel
from adv.utils import get_logger, trades_loss, classify, classify_ensemble, get_acc, get_shannon_entropy, get_logger, quantize
from adv.pgd_attack import PGDAttack
from adv.pgd_attack_transformation import PGDTransformAttack

with open('test_mnist.yml', 'r') as stream:
    config = yaml.safe_load(stream)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = config['meta']['gpu_id']

device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = config['test']['batch_size']

def load_ensemble_model(name):
    path = os.path.join('random_models', name)
    model_files = os.listdir(path)
    single_models = []
    for file in model_files:
        rand_params = config['rand']
        net = RandModel(BatchNormModel().to(device), rand_params)
        net = net.eval().to(device)
        net.load_state_dict(torch.load(os.path.join(path, file)))
        single_models.append(net)
    return EnsembleModel(single_models).to(device)

def main():
    (x_train, y_train), (_, _), (x_test, y_test) = load_mnist_all(
        data_dir=config['meta']['data_path'], val_size=0.1, shuffle=True,
        seed=1000)

    log = get_logger('test_mnist_trans_ensemble_no_at', 'test_mnist')

    ensemble = load_ensemble_model('saved_models')

    num_classes = 10
    num_test_samples = config['test']['num_test_samples']
    if config['pgd']['quant']:
        y_pred = classify_ensemble(ensemble, quantize(
            x_test[:num_test_samples]), num_classes=num_classes)
    else:
        y_pred = classify_ensemble(
            ensemble, x_test[:num_test_samples], num_classes=num_classes)

    acc = get_acc(y_pred, y_test[:num_test_samples])
    log.info('Clean acc: %.4f', acc)

    entropy = get_shannon_entropy(y_pred)
    log.info('Average entropy: %.4f', torch.mean(entropy))
    log.info('Median entropy: %.4f', torch.median(entropy))
    log.info('Max entropy: %.4f', torch.max(entropy))
    log.info('Min entropy: %.4f', torch.min(entropy))

    log.info('Starting ensemble PGD attack...')
    attack = PGDTransformAttack(ensemble, x_train, y_train)
    x_adv = attack(x_test[:num_test_samples], y_test[:num_test_samples], batch_size=batch_size, **config['pgd'])
    y_pred = classify_ensemble(ensemble, x_adv, num_classes=num_classes)

    adv_acc = get_acc(y_pred, y_test[:num_test_samples])
    log.info('Adv acc: %.4f', adv_acc)

    entropy = get_shannon_entropy(y_pred)
    log.info('Average entropy: %.4f', torch.mean(entropy))
    log.info('Median entropy: %.4f', torch.median(entropy))
    log.info('Max entropy: %.4f', torch.max(entropy))
    log.info('Min entropy: %.4f', torch.min(entropy))


if __name__ == '__main__':
    main()
