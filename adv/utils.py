'''Collection of utility and helper functions'''
import logging

import numpy as np
import torch
import torch.nn.functional as F


def get_logger(name, logger_name=None):
    # Get logger
    if logger_name is None:
        logger_name = name
    log_file = name + '.log'
    log = logging.getLogger(logger_name)
    log.setLevel(logging.DEBUG)
    # Create formatter and add it to the handlers
    formatter = logging.Formatter(
        '[%(levelname)s %(asctime)s %(name)s] %(message)s')
    # Create file handler
    fh = logging.FileHandler(log_file, mode='w')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    log.addHandler(fh)
    return log

def classify(net, x, batch_size=200, num_classes=10):
    """Classify <x> with <net>."""
    with torch.no_grad():
        y_pred = torch.zeros((x.size(0), num_classes))
        for i in range(int(np.ceil(x.size(0) / batch_size))):
            begin = i * batch_size
            end = (i + 1) * batch_size
            y_pred[begin:end] = net(x[begin:end].to('cuda'))
    return y_pred

def classify_ensemble(ensemble, x, batch_size=200, num_classes=10, method='aggregate_vote'):
    with torch.no_grad():
        y_preds = []
        for net in ensemble.models:
            y_pred = torch.zeros((x.size(0), num_classes)).to('cuda')
            for i in range(int(np.ceil(x.size(0) / batch_size))):
                begin = i * batch_size
                end = (i + 1) * batch_size
                y_pred[begin:end] = net(x[begin:end].to('cuda'))
            y_preds.append(y_pred)
        if method == 'aggregate_vote':
            return np.sum(y_preds, axis=0)
        else:
            raise NotImplementedError

def majority_vote(y_preds):
    votes = np.array([np.argmax(pred) for pred in y_preds])
    vote_vec, count_vec = np.unique(votes, return_counts=True)
    return vote_vec[np.argmax(count_vec)]

def get_acc(y_pred, y_test):
    """Compute accuracy based on network output (logits)."""
    return (y_pred.argmax(1) == y_test.to(y_pred.device)).float().mean().item()


def quantize(x, levels=16):
    """Quantization function from Qai et al. 2018 (CAT 2018)."""
    quant = torch.zeros_like(x)
    for i in range(1, levels):
        quant += (x >= i / levels).float()
    return quant / (levels - 1)


def trades_loss(logits, targets, params):
    """Compute loss for TRADES."""
    loss_natural = F.cross_entropy(logits[0], targets)
    loss_robust = F.kl_div(F.log_softmax(logits[1], dim=1),
                           F.softmax(logits[0], dim=1))
    return loss_natural + params['beta'] * loss_robust
