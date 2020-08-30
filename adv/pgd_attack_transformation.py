'''Implement PGD attack for evaluating robustness of neural networks.'''
import numpy as np
import torch
import torch.nn.functional as F

from .utils import classify, quantize


def best_other_class(logits, exclude):
    """Returns the index of the largest logit, ignoring the class that
    is passed as `exclude`."""
    y_onehot = torch.zeros_like(logits)
    y_onehot.scatter_(1, exclude, 1)
    # make logits that we want to exclude a large negative number
    other_logits = logits - y_onehot * 1e9
    return other_logits.max(1)[0]


class PGDAttack(object):
    """Implement PGD attack with additional options."""

    def __init__(self, net, x_train=None, y_train=None):
        self.net = net
        self.x_train = x_train
        self.y_train = y_train
        self.device = next(net.parameters()).device

        if x_train is not None and y_train is not None:
            num_train = x_train.size(0)
            with torch.no_grad():
                self.y_pred = torch.zeros(
                    num_train, device='cpu', dtype=torch.long)
                batch_size = 200
                num_batches = np.ceil(num_train / batch_size).astype(np.int32)
                for i in range(num_batches):
                    begin = i * batch_size
                    end = (i + 1) * batch_size
                    y_pred = net(x_train[begin:end].to('cuda')).argmax(1).cpu()
                    self.y_pred[begin:end] = y_pred

    def __call__(self, x_orig, label, batch_size=100, **kwargs):

        x_adv = torch.zeros_like(x_orig)
        num_batches = int(np.ceil(x_orig.size(0) / batch_size))

        for i in range(num_batches):
            begin = i * batch_size
            end = (i + 1) * batch_size
            x_adv[begin:end] = self.attack_batch(
                x_orig[begin:end], label[begin:end], **kwargs)
        return x_adv

    def attack_batch(self, x_orig, label, p='inf', targeted=False,
                     epsilon=0.03, step_size=0.01, num_steps=1000,
                     num_restarts=0, loss_func='ce', init_mode=1,
                     random_start=True, clip=True, quant=False, **kwargs):

        x_adv = x_orig.clone()
        x_orig = x_orig.to(self.device)
        label = label.to(self.device)
        batch_size = x_orig.size(0)
        zero = torch.zeros(1, device=self.device) + 1e-6

        num_draws = 3

        if p not in ['2', 'inf']:
            raise NotImplementedError('Norm not implemented (only 2 or inf)!')

        # initialize starting point of PGD according to specified init_mode
        if init_mode == 1:
            # init w/ original point
            x_init = x_orig.clone()
        elif init_mode == 2:
            # init w/ nearest training sample that has a different label
            x_top1 = self.find_neighbor_diff_class(x_orig, label, p)
            x_init = self.project_eps(x_orig, x_top1, p, epsilon)
        elif init_mode == 3:
            # init w/ k nearest training samples that have a different label
            x_topk = self.find_kth_neighbor_diff_class(
                x_orig, label, p, num_restarts + 1)
        else:
            raise ValueError('Invalid init_mode (only 1, 2, or 3)!')

        for i in range(num_restarts + 1):

            # for init_mode = 3, set new starting point at every restart
            if init_mode == 3:
                x_init = self.project_eps(x_orig, x_topk[i], p, epsilon)

            # add noise to the starting point if specified
            if random_start:
                if p == '2':
                    noise = torch.zeros_like(x_orig).normal_(0, epsilon)
                    noise = torch.renorm(noise.view(batch_size, -1),
                                         2, 0, epsilon).view(x_orig.size())
                elif p == 'inf':
                    noise = torch.zeros_like(x_orig).uniform_(
                        - epsilon, epsilon)
                x = x_init + noise
            else:
                x = x_init

            # clip to [0, 1]
            if clip:
                x.clamp_(0, 1)

            for _ in range(num_steps):

                # compute loss and gradients
                x.requires_grad_()
                with torch.enable_grad():
                    logits = self.net(x)
                    if num_draws > 1:
                        sf_logits = torch.nn.Softmax(dim=2)(logits)
                        avg_sf_per_batch = torch.mean(sf_logits, dim=1)
                        avg_logits = torch.log(avg_sf_per_batch)
                        if loss_func == 'ce':
                            loss = F.cross_entropy(avg_logits, label, reduction='sum')
                        elif loss_func == 'hinge':
                            other = best_other_class(avg_logits, label.unsqueeze(1))
                            loss = other - torch.gather(
                                avg_logits, 1, label.unsqueeze(1)).squeeze()
                            loss = torch.min(zero, loss).sum()
                    else:
                        if loss_func == 'ce':
                            loss = F.cross_entropy(logits, label, reduction='sum')
                        elif loss_func == 'hinge':
                            other = best_other_class(logits, label.unsqueeze(1))
                            loss = other - torch.gather(
                                logits, 1, label.unsqueeze(1)).squeeze()
                            loss = torch.min(zero, loss).sum()
                #print(loss)

                with torch.no_grad():
                    grad = torch.autograd.grad(loss, x)[0].detach()
                    if targeted:
                        grad *= -1

                    if p == '2':
                        grad_norm = torch.max(
                            grad.view(batch_size, -1).norm(2, 1), zero)
                        if x.dim() == 4:
                            step = grad / grad_norm.view(batch_size, 1, 1, 1)
                        else:
                            step = grad / grad_norm.view(batch_size, 1)
                    elif p == 'inf':
                        step = torch.sign(grad)

                    x = x.detach() + step_size * step

                    # clip to epsilon ball
                    x = self.project_eps(x_orig, x, p, epsilon)
                    # clip to [0, 1]
                    if clip:
                        x.clamp_(0, 1)

            if quant:
                x = quantize(x.detach())
            if not targeted:
                idx = np.where(
                    (self.net(x).argmax(1) != label).cpu().numpy())[0]
            else:
                idx = np.where(
                    (self.net(x).argmax(1) == label).cpu().numpy())[0]

            for j in idx:
                x_adv[j] = x[j]

            # TODO: can further optimize by removing already successful samples

        return x_adv

    @staticmethod
    def project_eps(x_orig, x_nn, p, epsilon):
        """Project <x_nn> onto an epsilon-ball around <x_orig>. The ball is
        specified by <p> and <epsilon>."""

        if p == '2':
            diff = (x_nn - x_orig).view(x_orig.size(0), -1).renorm(
                2, 0, epsilon)
            x_init = diff.view(x_orig.size()) + x_orig
        elif p == 'inf':
            x_init = (x_nn - x_orig).clamp(- epsilon, epsilon) + x_orig
        else:
            raise NotImplementedError('Specified lp-norm is not implemented.')
        return x_init

    def find_neighbor_diff_class(self, x, label, p):
        """Find the nearest training sample to x that has a different label"""

        nn = torch.zeros((x.size(0), ), dtype=torch.long)
        norm = 2 if p == '2' else np.inf

        for i in range(x.size(0)):
            dist = (x[i].cpu() - self.x_train).view(
                self.x_train.size(0), -1).norm(norm, 1)
            # we want to exclude samples that are classified to the
            # same label as x_orig
            ind = np.where(self.y_pred == label[i].cpu())[0]
            dist[ind] += 1e9
            nn[i] = dist.argmin()
        return self.x_train[nn].to(x.device)

    def find_kth_neighbor_diff_class(self, x, label, p, k):
        """Find k-th nearest training sample to x that has a different label"""

        x_topk = torch.zeros((k, ) + x.size())
        norm = 2 if p == '2' else np.inf

        for i in range(x.size(0)):
            dist = (x[i].cpu() - self.x_train).view(
                self.x_train.size(0), -1).norm(norm, 1)
            # we want to exclude samples that are classified to the
            # same label as x_orig
            ind = np.where(self.y_pred == label[i].cpu())[0]
            dist[ind] += 1e9
            topk = torch.topk(dist, k, largest=False)[1]
            x_topk[:, i] = self.x_train[topk]

        return x_topk.to(x.device)

# =========================================================================== #
#                        Helper Functions for PGD+                            #
# =========================================================================== #


def pgdp(net, x_train, y_train, x_test, y_test, batch_size, params,
         num_classes=10):
    """A helper function for running PGD+."""
    attack = PGDAttack(net, x_train, y_train)
    idx_best = np.ones(len(x_test))
    for loss in ['ce', 'hinge']:
        for init_mode in [1, 3]:
            params['loss'] = loss
            params['init_mode'] = init_mode
            x_adv = attack(x_test, y_test, batch_size=batch_size, **params)
            y_pred = classify(net, x_adv, num_classes=num_classes)
            idx_best *= (y_pred.argmax(1).cpu() == y_test).numpy()
    return (idx_best > 0).mean()


# =========================================================================== #
#                        Attack on Spheres dataset                            #
# =========================================================================== #


class PGDAttackSpheres(object):
    """
    Run PGD attack with L2 perturbation on Sphere Dataset
    """

    def __call__(self, net, x_orig, label, radii=(1, 1.3), centers=(0, 0),
                 epsilon=0.03, step_size=0.01, max_iterations=1000,
                 num_restarts=0, batch_size=100, eps_proj=False):

        x_adv = torch.zeros_like(x_orig)
        num_batches = x_orig.size(0) // batch_size
        i = -1

        for i in range(num_batches):
            begin = i * batch_size
            end = (i + 1) * batch_size
            x_adv[begin:end] = self.attack_batch(
                net, x_orig[begin:end], label[begin:end], radii=radii,
                centers=centers, epsilon=epsilon, step_size=step_size,
                max_iterations=max_iterations, num_restarts=num_restarts,
                eps_proj=eps_proj)

        if x_orig.size(0) % batch_size != 0:
            begin = (i + 1) * batch_size
            end = (i + 2) * batch_size
            x_adv[begin:end] = self.attack_batch(
                net, x_orig[begin:end], label[begin:end], radii=radii,
                centers=centers, epsilon=epsilon, step_size=step_size,
                max_iterations=max_iterations, num_restarts=num_restarts,
                eps_proj=eps_proj)

        return x_adv

    def attack_batch(self, net, x_orig, label, radii=(1, 1.3), centers=(0, 0),
                     epsilon=0.03, step_size=0.01, max_iterations=100,
                     num_restarts=0, eps_proj=False):

        x_adv = torch.zeros_like(x_orig)
        x_orig = x_orig.to('cuda')
        label = label.to('cuda').float().unsqueeze(1)
        best_dist = torch.zeros(x_orig.size(0), device='cuda') + 1e9
        # zero = torch.tensor(-1e5).cuda()
        zero = torch.tensor(0.).cuda()

        for i in range(num_restarts + 1):

            # For any additional restart, uniform noise is added
            # if i > 0:
            #     x = x_orig + \
            #         torch.zeros_like(x_orig).uniform_(-epsilon, epsilon)
            #     # Project back to spheres
            #     x = self.normalize(x, label, radii, centers)
            # else:
            #     x = x_orig.clone()
            noise = torch.zeros_like(x_orig).normal_(0, step_size)
            x = x_orig + torch.renorm(noise, 2, 0, epsilon)

            for j in range(max_iterations):
                x.requires_grad_()
                with torch.enable_grad():
                    logits = net(x)
                    # loss = F.cross_entropy(logits, label, reduction='sum')
                    # logits can be large and saturates sigmoid
                    # loss = F.binary_cross_entropy_with_logits(
                    #     logits, label, reduction='sum')
                    loss = - torch.max(zero, (2 * label - 1) * logits).sum()

                grad = torch.autograd.grad(loss, x)[0].detach()
                # x = x.detach() + step_size * F.normalize(grad, 2, 1)

                grad_norm = torch.max(
                    grad.view(x.size(0), -1).norm(2, 1), torch.tensor(1e-9).cuda())
                delta = step_size * grad / grad_norm.view(x.size(0), 1)
                x = x.detach() + delta

                # Project to epsilon ball if specified
                if eps_proj:
                    delta = torch.renorm((x - x_orig), 2, 0, epsilon)
                    x = x_orig + delta

                # Project back to spheres
                x = self.normalize(x, label, radii, centers)

                # Keep adv examples with smallest perturbation
                y_pred = (net(x) >= 0).long().squeeze(1)
                idx = (y_pred != label.long().squeeze(1)).cpu().numpy()
                dist = torch.norm(x - x_orig, 2, 1)
                idx *= (dist < best_dist).cpu().numpy()
                idx = np.where(idx)[0]
                x_adv[idx] = x[idx].cpu()
                best_dist[idx] = dist[idx]

            # TODO: can further optimize by removing already successful samples

        return x_adv

    def normalize(self, x, label, radii, centers):
        x -= (1 - label) * centers[0]
        x -= label * centers[1]
        x = F.normalize(x, 2, 1)
        x *= (radii[0] + label * (radii[1] - radii[0]))
        x += (1 - label) * centers[0]
        x += label * centers[1]
        return x
