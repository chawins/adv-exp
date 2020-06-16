import logging

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

INF = 1e15


class CWL2AttackSpheres(object):
    """
    """

    def __call__(self, net, x_orig, label, radii, binary_search_steps=10,
                 max_iterations=1000, confidence=0, learning_rate=1e-1,
                 initial_const=1, abort_early=True, verbose=True):
        """
        x_orig is tensor (requires_grad=False)
        """

        label = label.view(-1, 1).to('cuda').float()
        batch_size = x_orig.size(0)
        x_orig = x_orig.to('cuda')
        x_adv = x_orig.clone()

        # declare tensors that keep track of constants and binary search
        const = torch.zeros((batch_size, ), device=x_orig.device)
        const += initial_const
        lower_bound = torch.zeros_like(const)
        upper_bound = torch.zeros_like(const) + INF
        best_l2dist = torch.zeros_like(const) + INF

        for binary_search_step in range(binary_search_steps):
            if binary_search_step == binary_search_steps - 1 and \
                    binary_search_steps >= 10:
                # in the last binary search step, use the upper_bound instead
                const = upper_bound

            delta = torch.zeros_like(x_adv, requires_grad=True)
            loss_at_previous_check = torch.zeros(1, device=x_orig.device) + INF

            # create a new optimizer
            optimizer = optim.Adam([delta], lr=learning_rate)

            for iteration in range(max_iterations):
                optimizer.zero_grad()
                x = F.normalize(x_orig + delta, 2, 1)
                x *= (radii[0] + label * (radii[1] - radii[0]))
                logits = net(x)
                loss, l2dist = self.loss_function(
                    x, label, logits, const, x_orig, confidence)
                loss.backward()
                optimizer.step()

                if iteration % (np.ceil(max_iterations / 10)) == 0 and verbose:
                    print('    step: %d; loss: %.3f; l2dist: %.3f' %
                          (iteration, loss.cpu().detach().numpy(),
                           l2dist.mean().cpu().detach().numpy()))

                if abort_early and iteration % (np.ceil(max_iterations / 10)) == 0:
                    # after each tenth of the iterations, check progress
                    if torch.gt(loss, .9999 * loss_at_previous_check):
                        break  # stop Adam if there has not been progress
                    loss_at_previous_check = loss

            with torch.no_grad():
                is_adv = self.check_adv(logits, label, confidence)

            for i in range(batch_size):
                if is_adv[i]:
                    # sucessfully find adv
                    upper_bound[i] = const[i]
                else:
                    # fail to find adv
                    lower_bound[i] = const[i]

            for i in range(batch_size):
                if upper_bound[i] == INF:
                    # exponential search if adv has not been found
                    const[i] *= 10
                else:
                    # binary search if adv has been found
                    const[i] = (lower_bound[i] + upper_bound[i]) / 2
                # only keep adv with smallest l2dist
                if is_adv[i] and best_l2dist[i] > l2dist[i]:
                    x_adv[i] = x[i]
                    best_l2dist[i] = l2dist[i]

            with torch.no_grad():
                logits = net(x_adv)
                is_adv = self.check_adv(logits, label, confidence)
            if verbose:
                print('binary step: %d; number of successful adv: %d/%d' %
                      (binary_search_step, is_adv.sum().cpu().numpy(), batch_size))

        return x_adv.cpu()

    @classmethod
    def check_adv(cls, logits, label, confidence):
        """return True if logits do not match with label (misclassification)"""
        return ((1 - 2 * label) * logits - confidence) >= 0

    @classmethod
    def loss_function(cls, x_adv, label, logits, const, x_orig, confidence):
        """Returns the loss and the gradient of the loss w.r.t. x,
        assuming that logits = model(x)."""

        adv_loss = (2 * label - 1) * logits
        adv_loss = torch.max(torch.zeros_like(adv_loss), adv_loss + confidence)

        l2dist = (x_adv - x_orig).norm(2, 1) ** 2
        total_loss = l2dist + const * adv_loss.squeeze()

        return total_loss.mean(), l2dist.sqrt()
