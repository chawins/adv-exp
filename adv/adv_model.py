'''Implement wrapper Module on Pytorch Module for adversarial training (AT).'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def best_other_class(logits, exclude):
    """Returns the index of the largest logit, ignoring the class that
    is passed as `exclude`."""
    y_onehot = torch.zeros_like(logits)
    y_onehot.scatter_(1, exclude, 1)
    # make logits that we want to exclude a large negative number
    other_logits = logits - y_onehot * 1e9
    return other_logits.max(1)[0]


class PGDModel(nn.Module):
    """
    code adapted from
    https://github.com/karandwivedi42/adversarial/blob/master/main.py
    """

    def __init__(self, basic_net, params):
        super(PGDModel, self).__init__()
        self.basic_net = basic_net
        self.params = params

    def parse_params(self, params):
        """Parse given parameters for AT."""
        p = params['p']
        step_size = params['step_size']
        epsilon = params['epsilon']
        loss_func = params['loss_func']
        gap = params['gap']
        clip = params['clip']
        if params['use_diff_rand_eps']:
            rand_eps = params['rand_eps']
        else:
            rand_eps = epsilon
        return (p, step_size, epsilon, loss_func, gap, clip, rand_eps)

    def cal_gap(self, x, inputs, targets, params=None):
        """Compute Frank-Wolfe optimality gap (see Wang et al. 2018)."""
        if not params:
            params = self.params
        p, _, epsilon, loss_func, _, _, _ = self.parse_params(params)
        self.basic_net.eval()
        x.requires_grad_()
        # calculate loss
        with torch.enable_grad():
            logits = self.basic_net(x)
            if loss_func == 'ce':
                loss = F.cross_entropy(logits, targets, reduction='sum')
            elif loss_func == 'hinge':
                other = best_other_class(logits, targets.unsqueeze(1))
                loss = other - \
                    torch.gather(logits, 1, targets.unsqueeze(1)).squeeze()
                # positive gap creates stronger adv
                loss = torch.min(torch.zeros_like(loss), loss).sum()
        grad = torch.autograd.grad(loss, x)[0].detach().view(x.size(0), -1)

        with torch.no_grad():
            if p == 'inf':
                grad_norm = epsilon * grad.abs().sum(1)
            elif p == '2':
                grad_norm = epsilon * grad.norm(2, 1)
            iprod = ((x - inputs).view(x.size(0), -1) * grad).sum(1)
            fosc = grad_norm - iprod

        return fosc

    def _compute_mask(self, x, inputs, logits, targets, grad, params,
                      is_train=True):
        """Compute mask on samples for early stopping."""

        with torch.no_grad():
            # compute specified threshold
            if params['early_stop'] and is_train:
                softmax = F.softmax(logits.detach(), dim=1)
                prob = torch.gather(
                    softmax, 1, targets.unsqueeze(1)).squeeze()
                other = best_other_class(softmax, targets.unsqueeze(1))
                mask = ((other - prob) <= params['gap']).float()
            elif params['use_fosc'] and is_train:
                batch_size = x.size(0)
                if params['p'] == 'inf':
                    # when p = inf, compute l1-norm of grad
                    grad_norm = grad.view(batch_size, -1).abs().sum(1)
                elif params['p'] == '2':
                    grad_norm = grad.view(batch_size, -1).norm(2, 1)
                iprod = ((x - inputs) * grad).view(batch_size, -1).sum(1)
                fosc = params['epsilon'] * grad_norm - iprod
                mask = (fosc >= params['fosc_thres']).float()
            else:
                mask = torch.ones(targets.size(0), device=logits.device)

        mask.requires_grad_(False)
        if x.dim() == 4:
            return mask.view(-1, 1, 1, 1)
        return mask.view(-1, 1)

    def forward(self, inputs, targets, adv=True, params=None, cal_gap=False,
                cal_gap_params=None):
        """Forward pass for finding adversarial examples for AT.
        There is also an option to compute Frank-Wolfe optimality gap (see
        Wang et al. 2018 for more detail).

        Args:
            inputs (torch.tensor): input samples
            targets (torch.tensor): ground-truth label
            adv (bool, optional): whether to use AT
            params (dict, optional): parameters for AT
            cal_gap (bool, optional): whether to compute FW gap
            cal_gap_params (dict, optional): parameters for computing FW gap

        Returns:
            logits (torch.tensor): logits output by self.basic_net
        """
        if not adv:
            return self.basic_net(inputs)
        if params is None:
            params = self.params
        if params['method'] == 'none':
            return self.basic_net(inputs)

        p, step_size, epsilon, loss_func, gap, clip, rand_eps = \
            self.parse_params(params)

        # set network to eval mode to remove some training behavior (e.g.
        # dropout, batch norm)
        is_train = self.basic_net.training
        self.basic_net.eval()
        x = inputs.clone()

        # compute logits of the clean samples for TRADES
        if loss_func == 'trades':
            logits_clean = self.basic_net(inputs.detach())
            softmax_clean = F.softmax(logits_clean.detach(), dim=1)

        # randomly initialize adversarial examples
        if params['random_start']:
            if p == 'inf':
                x = x + torch.zeros_like(x).uniform_(- rand_eps, rand_eps)
            elif p == '2':
                noise = torch.zeros_like(x).normal_(0, 1).view(x.size(0), -1)
                x += torch.renorm(noise, 2, 0, rand_eps).view(x.size())
            if clip:
                x = torch.clamp(x, 0, 1)

        for _ in range(params['num_steps']):

            x.requires_grad_()
            with torch.enable_grad():
                # get logits from the current samples
                logits = self.basic_net(x)

                # compute loss
                if loss_func == 'ce' or not is_train:
                    loss = F.cross_entropy(logits, targets, reduction='sum')
                elif loss_func == 'clipped_ce':
                    logsoftmax = torch.clamp(
                        F.log_softmax(logits, dim=1), np.log(gap), 0)
                    loss = F.nll_loss(logsoftmax, targets, reduction='sum')
                elif loss_func == 'hinge':
                    other = best_other_class(logits, targets.unsqueeze(1))
                    loss = other - \
                        torch.gather(logits, 1, targets.unsqueeze(1)).squeeze()
                    # Positive gap creates stronger adv
                    loss = torch.min(torch.tensor(gap).cuda(), loss).sum()
                elif loss_func == 'trades':
                    log_softmax_adv = F.log_softmax(logits, dim=1)
                    loss = F.kl_div(
                        log_softmax_adv, softmax_clean, reduction='sum')
                else:
                    raise NotImplementedError('loss function not implemented.')

            # compute gradients
            grad = torch.autograd.grad(loss, x)[0].detach()

            # compute mask for updating perturbation (only used by ATES and
            # Dynamic AT). <mask> is all ones otherwise.
            mask = self._compute_mask(
                x, inputs, logits, targets, grad, params, is_train=is_train)
            if mask.sum() == 0:
                break

            # compute updates on the current samples
            if p == 'inf':
                x = x.detach() + step_size * mask * torch.sign(grad)
                # projection step to l-inf ball
                x = torch.min(torch.max(x, inputs.detach() - epsilon),
                              inputs.detach() + epsilon)
            elif p == '2':
                grad_norm = torch.max(
                    grad.view(x.size(0), -1).norm(2, 1),
                    torch.tensor(1e-9).cuda())
                if inputs.dim() == 4:
                    delta = step_size * grad / \
                        grad_norm.view(x.size(0), 1, 1, 1)
                else:
                    delta = step_size * grad / grad_norm.view(x.size(0), 1)

                # take PGD step
                x = x.detach() + mask * delta
                # project back to epsilon ball (delta is redefined here to
                # overall perturbation not a single step)
                delta = torch.renorm((x - inputs.detach()).view(x.size(0), -1),
                                     2, 0, epsilon).view(x.size())
                x = inputs.detach() + delta
            else:
                raise NotImplementedError('specified lp-norm not implemented.')
            # clip samples to [0, 1] if specified
            if clip:
                x = torch.clamp(x, 0, 1)

        # only used for computing FOSC
        if cal_gap:
            fosc = self.cal_gap(x, inputs, targets, params=cal_gap_params)
            self.basic_net.train(is_train)
            return x, fosc, (x - inputs).view(x.size(0), -1).abs().max(1)[0]

        # set the network to its original state (train or eval)
        self.basic_net.train(is_train)
        return self.basic_net(x)


# =========================================================================== #


class FGSMModel(nn.Module):
    """
    """

    def __init__(self, basic_net, params):
        super(FGSMModel, self).__init__()
        self.basic_net = basic_net
        self.params = params

    def parse_params(self, params):
        p = params['p']
        epsilon = params['epsilon']
        loss_func = params['loss_func']
        gap = params['gap']
        clip = params['clip']
        if params['use_diff_rand_eps']:
            rand_eps = params['rand_eps']
        else:
            rand_eps = epsilon
        return (p, epsilon, loss_func, gap, clip, rand_eps)

    def cal_gap(self, x, inputs, targets, params=None):
        if not params:
            params = self.params
        p, epsilon, loss_func, _, _, _ = self.parse_params(params)
        self.basic_net.eval()
        x.requires_grad_()
        # Calculate loss
        with torch.enable_grad():
            logits = self.basic_net(x)
            if loss_func == 'ce':
                loss = F.cross_entropy(logits, targets, reduction='sum')
            elif loss_func == 'hinge':
                other = best_other_class(logits, targets.unsqueeze(1))
                loss = other - \
                    torch.gather(logits, 1, targets.unsqueeze(1)).squeeze()
                # Positive gap creates stronger adv
                loss = torch.min(torch.zeros_like(loss), loss).sum()
        grad = torch.autograd.grad(loss, x)[0].detach().view(x.size(0), -1)

        with torch.no_grad():
            if p == 'inf':
                grad_norm = epsilon * grad.abs().sum(1)
            elif p == '2':
                grad_norm = epsilon * grad.norm(2, 1)
            ip = ((x - inputs).view(x.size(0), -1) * grad).sum(1)
            c = grad_norm - ip

        return c, grad

    def forward(self, inputs, targets, adv=True, params=None, cal_gap=False,
                cal_gap_params=None):
        if not adv:
            return self.basic_net(inputs)
        if not params:
            params = self.params
        p, epsilon, loss_func, gap, clip, rand_eps = \
            self.parse_params(params)

        # set network to eval mode to remove some training behavior (e.g.
        # dropout, batch norm)
        is_train = self.basic_net.training
        self.basic_net.eval()
        x = inputs.clone()

        # Wong et al. find it beneficial to increase epsilon during training by
        # a factor of 0.25 for the best robustness at the original epsilon
        if is_train:
            epsilon *= 1.25

        if params['random_start']:
            if p == 'inf':
                x = x + torch.zeros_like(x).uniform_(- rand_eps, rand_eps)
            elif p == '2':
                noise = torch.zeros_like(x).normal_(0, 1).view(x.size(0), -1)
                x += torch.renorm(noise, 2, 0, rand_eps).view(x.size())
            if clip:
                x = torch.clamp(x, 0, 1)

        x.requires_grad_()
        with torch.enable_grad():
            logits = self.basic_net(x)

        # determine step size for early stopping
        if params['early_stop']:
            softmax = F.softmax(logits, dim=1)
            prob = torch.gather(
                softmax, 1, targets.unsqueeze(1)).squeeze()
            other = best_other_class(softmax, targets.unsqueeze(1))
            loss = other - prob

            # compute gradients
            grad = torch.autograd.grad(loss.sum(), x)[0].detach()

            delta_f = F.relu(params['gap'] - loss)
            if p == 'inf':
                grad_norm = grad.view(x.size(0), -1).abs().sum(1)
            elif p == '2':
                grad_norm = grad.view(x.size(0), -1).norm(2, 1)
            grad_norm += 1e-9
            step_size = delta_f / grad_norm
            step_size = step_size.clamp(0, epsilon).view(-1, 1, 1, 1)
        else:
            if loss_func == 'ce':
                loss = F.cross_entropy(logits, targets, reduction='sum')
            elif loss_func == 'clipped_ce':
                logsoftmax = torch.clamp(
                    F.log_softmax(logits, dim=1), np.log(gap), 0)
                loss = F.nll_loss(logsoftmax, targets, reduction='sum')
            elif loss_func == 'hinge':
                other = best_other_class(logits, targets.unsqueeze(1))
                loss = other - \
                    torch.gather(logits, 1, targets.unsqueeze(1)).squeeze()
                # Positive gap creates stronger adv
                loss = torch.min(torch.tensor(gap).cuda(), loss).sum()
            else:
                raise NotImplementedError('loss function not implemented.')

            # compute gradients
            grad = torch.autograd.grad(loss, x)[0].detach()

            step_size = epsilon

        # compute the update
        if p == 'inf':
            x = x.detach() + step_size * torch.sign(grad)
            x = torch.min(torch.max(x, inputs.detach() - epsilon),
                          inputs.detach() + epsilon)
        elif p == '2':
            if inputs.dim() == 4:
                delta = step_size * grad / grad_norm.view(x.size(0), 1, 1, 1)
            else:
                delta = step_size * grad / grad_norm.view(x.size(0), 1)
            # Take PGD step
            x = x.detach() + delta
            # Project back to epsilon ball (delta is redefined here to
            # overall perturbation not a single step)
            delta = torch.renorm((x - inputs.detach()).view(x.size(0), -1),
                                 2, 0, epsilon).view(x.size())
            x = inputs.detach() + delta
        if clip:
            x = torch.clamp(x, 0, 1)

        # only used for computing FOSC
        if cal_gap:
            c, grad = self.cal_gap(x, inputs, targets, params=cal_gap_params)
            self.basic_net.train(is_train)
            return x, c, grad

        self.basic_net.train(is_train)
        return self.basic_net(x)


# =========================================================================== #
