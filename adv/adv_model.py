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
        if not params:
            params = self.params
        p, _, epsilon, loss_func, _, _, _ = self.parse_params(params)
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
            iprod = ((x - inputs).view(x.size(0), -1) * grad).sum(1)
            fosc = grad_norm - iprod

        return fosc

    def _compute_mask(self, x, inputs, logits, targets, grad, params,
                      is_train=True):

        with torch.no_grad():
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

        if loss_func == 'trades':
            logits_clean = self.basic_net(inputs.detach())
            softmax_clean = F.softmax(logits_clean.detach(), dim=1)

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
                logits = self.basic_net(x)

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

            # compute mask for updating perturbation
            mask = self._compute_mask(
                x, inputs, logits, targets, grad, params, is_train=is_train)
            if mask.sum() == 0:
                break

            # compute the update
            if p == 'inf':
                x = x.detach() + step_size * mask * torch.sign(grad)
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

                # Take PGD step
                x = x.detach() + mask * delta
                # Project back to epsilon ball (delta is redefined here to
                # overall perturbation not a single step)
                delta = torch.renorm((x - inputs.detach()).view(x.size(0), -1),
                                     2, 0, epsilon).view(x.size())
                x = inputs.detach() + delta
            else:
                raise NotImplementedError('specified lp-norm not implemented.')
            if clip:
                x = torch.clamp(x, 0, 1)

        # only used for computing FOSC
        if cal_gap:
            fosc = self.cal_gap(x, inputs, targets, params=cal_gap_params)
            self.basic_net.train(is_train)
            return x, fosc, (x - inputs).view(x.size(0), -1).abs().max(1)[0]

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


class PGDL2ModelSpheres(nn.Module):
    """
    code adapted from
    https://github.com/karandwivedi42/adversarial/blob/master/main.py
    """

    def __init__(self, basic_net, config):
        super(PGDL2ModelSpheres, self).__init__()
        self.basic_net = basic_net
        self.rand = config['random_start']
        self.step_size = config['step_size']
        self.epsilon = config['epsilon']
        self.num_steps = config['num_steps']
        self.radii = config['radii']
        self.centers = config['centers']
        self.loss_func = config['loss_func']
        assert self.loss_func in ['ce', 'bce', 'hinge', 'linear', 'sigmoid'], \
            'Only \'ce\' and \'bce\' are supported for now.'
        self.gap = torch.tensor(config['gap']).float().cuda()
        self.zero = torch.tensor(1e-9).cuda()

    def forward(self, inputs, targets, attack=False, train=False):
        if not attack:
            return self.basic_net(inputs)

        self.basic_net.eval()
        x = inputs.detach()
        tg = targets.float().unsqueeze(1)

        # Add random noise to the input at the beginning if specified
        if self.rand:
            # x = x + torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)
            noise = torch.zeros_like(x).normal_(
                0, self.step_size).view(x.size(0), -1)
            # Make sure the noise has norm at most epsilon
            x += torch.renorm(noise, 2, 0, self.epsilon).view(x.size())

        for _ in range(self.num_steps):
            x.requires_grad_()
            # Calculate loss
            with torch.enable_grad():
                logits = self.basic_net(x)
                if self.loss_func == 'ce':
                    loss = F.cross_entropy(logits, targets, reduction='sum')
                elif self.loss_func == 'bce':
                    loss = F.binary_cross_entropy_with_logits(
                        logits, tg, reduction='sum')
                elif self.loss_func == 'hinge':
                    # Positive gap makes adv weaker
                    loss = - torch.max(self.gap, (2 * tg - 1) * logits).sum()
                elif self.loss_func == 'linear':
                    loss = ((1 - 2 * tg) * logits).sum()
                elif self.loss_func == 'sigmoid':
                    f = 10 * (2 * tg - 1) * logits
                    loss = torch.sum(1 / (1 + torch.exp(f)))

            # Calculate gradients
            grad = torch.autograd.grad(loss, x)[0].detach()
            # Normalize gradients
            grad_norm = torch.max(
                grad.view(x.size(0), -1).norm(2, 1), self.zero)
            if inputs.dim() == 4:
                delta = self.step_size * grad / \
                    grad_norm.view(x.size(0), 1, 1, 1)
            else:
                delta = self.step_size * grad / grad_norm.view(x.size(0), 1)

            print(delta.norm(2, 1))

            # Take PGD step
            x = x.detach() + delta
            # x = torch.min(torch.max(x, inputs - self.epsilon),
            #               inputs + self.epsilon)
            # Project back to epsilon ball (delta is redefined here to overall
            # perturbation not a single step)
            delta = torch.renorm((x - inputs.detach()).view(x.size(0), -1),
                                 2, 0, self.epsilon).view(x.size())
            x = inputs.detach() + delta

            # Project the perturbed samples back to the spheres if specified
            if self.radii is not None and self.centers is not None:
                x -= (1 - tg) * self.centers[0]
                x -= tg * self.centers[1]
                x = F.normalize(x, 2, 1)
                x *= (self.radii[0] + tg * (self.radii[1] - self.radii[0]))
                x += (1 - tg) * self.centers[0]
                x += tg * self.centers[1]

        if train:
            self.basic_net.train()
        return self.basic_net(x)


# =========================================================================== #
