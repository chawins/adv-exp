from __future__ import print_function

import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml

from adv.dataset_utils import load_mnist_all
from adv.mnist_model import BasicModel, BatchNormModel, EnsembleModel
from adv.utils import get_logger, trades_loss, classify, classify_ensemble, get_acc, get_logger, quantize

with open('test_mnist.yml', 'r') as stream:
    config = yaml.safe_load(stream)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = config['meta']['gpu_id']

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_ensemble_model(name):
    path = os.path.join('saved_models', name)
    model_files = os.listdir(path)
    single_models = []
    for file in model_files:
        net = BatchNormModel()
        net = net.eval().to(device)
        net.load_state_dict(torch.load(os.path.join(path, file)))
        single_models.append(net)
    return EnsembleModel(single_models)

def main():
    (x_train, y_train), (_, _), (x_test, y_test) = load_mnist_all(
        data_dir=config['meta']['data_path'], val_size=0.1, shuffle=True,
        seed=1000)

    log = get_logger('test_mnist_ensemble', 'test_mnist')

    ensemble = load_ensemble_model('mnist_ensemble')

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


if __name__ == '__main__':
    main()
