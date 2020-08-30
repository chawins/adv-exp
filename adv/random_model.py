'''Implement wrapper Module on Pytorch Module for random transformation.'''
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.augmentation import (RandomErasing, RandomGrayscale,
                                 RandomHorizontalFlip, RandomMotionBlur)
from kornia.enhance.adjust import (adjust_brightness, adjust_contrast,
                                 adjust_gamma, adjust_hue, adjust_saturation)
# from kornia.filters.blur import (box_blur, gaussian_blur2d, median_blur,
#                                  motion_blur)
from kornia.filters import (BoxBlur, GaussianBlur2d, Laplacian, MedianBlur,
                            MotionBlur, Sobel)

from .utils import normalize


class RandModel(nn.Module):
    """
    """

    def __init__(self, basic_net, params):
        super(RandModel, self).__init__()
        self.basic_net = basic_net
        self.params = params

    def get_basic_net(self):
        return self.basic_net

    def forward(self, inputs, rand=True, num_draws=None, params=None,
                return_img=False):
        """

        Args:
            inputs (torch.tensor): input samples
            targets (torch.tensor): ground-truth label

        Returns:
            logits (torch.tensor): logits output by self.basic_net
        """
        if params is None:
            params = self.params
        if not rand:
            return self.basic_net(inputs)
        if num_draws is None:
            num_draws = params['num_draws']
        if params['seed'] is not None:
            np.random.seed(params['seed'])
            torch.manual_seed(params['seed'])
        batch_size = inputs.size(0)
        num_total = batch_size * num_draws

        # TODO: implement random ordering of transformations. This may require
        # sub-batch + batch to obtain different behavior. This can also make
        # blurring filter more efficient as well.

        if 'erase' in params['transforms']:
            erase = RandomErasing(
                1.0, params['erase']['scale'], params['erase']['ratio'])
        if 'hflip' in params['transforms']:
            hflip = RandomHorizontalFlip(params['hflip']['p'])
        if 'grayscale' in params['transforms']:
            grayscale = RandomGrayscale(params['grayscale']['p'])
        if 'boxblur' in params['transforms']:
            kernel_size = params['boxblur']['kernel_size']
            boxblur = BoxBlur((kernel_size, kernel_size))
        if 'gaussblur' in params['transforms']:
            kernel_size = params['gaussblur']['kernel_size']
            sigma = params['gaussblur']['sigma']
            gaussblur = GaussianBlur2d((kernel_size, kernel_size),
                                       (sigma, sigma))
        if 'medblur' in params['transforms']:
            kernel_size = params['medblur']['kernel_size']
            medblur = MedianBlur((kernel_size, kernel_size))
        if 'motionblur' in params['transforms']:
            motionblur = RandomMotionBlur(params['motionblur']['kernel_size'],
                                          params['motionblur']['angle'],
                                          params['motionblur']['direction'])
        if 'laplacian' in params['transforms']:
            laplacian = Laplacian(params['laplacian']['kernel_size'])
        if 'sobel' in params['transforms']:
            sobel = Sobel()

        x = inputs.repeat(num_draws, 1, 1, 1, 1)
        # reshape <x> to a batch of <num_total> samples
        x = x.view((num_total, ) + inputs.size()[1:])
        theta = torch.eye(3, device=x.device).repeat(num_total, 1, 1)

        for tf in params['transforms']:
            if tf == 'normal':
                x += torch.zeros_like(x).normal_(
                    params['normal']['mean'], params['normal']['std']).detach()
                # use separate clipping after adding noise as other transforms
                # may require [0, 1] range
                if params['normal']['clip'] is not None:
                    x.clamp_(params['normal']['clip'][0],
                             params['normal']['clip'][1])
            elif tf == 'uniform':
                rnge = params['uniform']['range']
                x += torch.zeros_like(x).uniform_(rnge[0], rnge[1]).detach()
            elif tf == 'scale':
                scale = torch.eye(3, device=x.device).repeat(num_total, 1, 1)
                alpha = params['scale']['alpha']
                if params['scale']['dist'] == 'uniform':
                    scale[:, 0, 0].uniform_(1 - alpha, 1 + alpha)
                    scale[:, 1, 1].uniform_(1 - alpha, 1 + alpha)
                elif params['scale']['dist'] == 'normal':
                    scale[:, 0, 0].normal_(0, alpha)
                    scale[:, 1, 1].normal_(0, alpha)
                theta = scale @ theta
            elif tf == 'rotate':
                rotate = torch.eye(3, device=x.device).repeat(num_total, 1, 1)
                angle = torch.zeros(num_total, device=x.device)
                alpha = params['rotate']['alpha'] / 180 * math.pi
                if params['rotate']['dist'] == 'uniform':
                    angle.uniform_(- alpha, alpha)
                elif params['rotate']['dist'] == 'normal':
                    angle.normal_(0, alpha)
                rotate[:, 0, 0] = torch.cos(angle)
                rotate[:, 1, 1] = torch.cos(angle)
                rotate[:, 0, 1] = torch.sin(angle)
                rotate[:, 1, 0] = - torch.sin(angle)
                theta = rotate @ theta
            elif tf == 'shear':
                shear = torch.eye(3, device=x.device).repeat(num_total, 1, 1)
                alpha = params['shear']['alpha']
                if params['shear']['dist'] == 'uniform':
                    shear[:, 0, 1].uniform_(- alpha, alpha)
                    shear[:, 1, 0].uniform_(- alpha, alpha)
                elif params['shear']['dist'] == 'normal':
                    shear[:, 0, 1].normal_(0, alpha)
                    shear[:, 1, 0].normal_(0, alpha)
                theta = shear @ theta
            elif tf == 'translate':
                translate = torch.eye(3, device=x.device).repeat(
                    num_total, 1, 1)
                alpha = params['translate']['alpha']
                if params['translate']['dist'] == 'uniform':
                    translate[:, 0, 2].uniform_(- alpha, alpha)
                    translate[:, 1, 2].uniform_(- alpha, alpha)
                elif params['translate']['dist'] == 'normal':
                    translate[:, 0, 2].normal_(0, alpha)
                    translate[:, 1, 2].normal_(0, alpha)
                theta = translate @ theta
            elif tf == 'affine':
                alpha = params['affine']['alpha']
                if params['affine']['dist'] == 'uniform':
                    perturb = torch.zeros_like(theta).uniform_(- alpha, alpha)
                elif params['affine']['dist'] == 'normal':
                    perturb = torch.zeros_like(theta).normal_(0, alpha)
                theta += perturb
            elif tf == 'erase':
                x = erase(x)
            elif tf == 'drop':
                x *= torch.zeros_like(x).bernoulli_(params['drop']['p'])
            elif tf == 'drop_pixel':
                mask = torch.zeros(
                    (num_total, 1, ) + x.size()[2:], device=x.device)
                x *= mask.bernoulli_(params['drop_pixel']['p'])
            elif tf == 'colorjitter':
                alpha = params['colorjitter']['alpha']
                brightness = torch.zeros(
                    num_total, device=x.device).uniform_(- alpha, alpha)
                contrast = torch.zeros(
                    num_total, device=x.device).uniform_(1 - alpha, 1 + alpha)
                saturation = torch.zeros(
                    num_total, device=x.device).uniform_(1 - alpha, 1 + alpha)
                # hue = torch.zeros(
                #     num_total, device=x.device).uniform_(1 - alpha, 1 + alpha)
                x = adjust_brightness(x, brightness)
                x = adjust_contrast(x, contrast)
                x = adjust_saturation(torch.clamp(x, 1e-9, 1e9), saturation)
                # x = adjust_hue(x, hue)
            elif tf == 'hflip':
                x = hflip(x)
            elif tf == 'gamma':
                gamma = torch.zeros(num_total, device=x.device)
                alpha = params['gamma']['alpha']
                if params['gamma']['dist'] == 'uniform':
                    gamma.uniform_(1 - alpha, 1 + alpha)
                elif params['gamma']['dist'] == 'normal':
                    gamma.normal_(1, alpha)
                gamma.clamp_(0, 10)
                x = adjust_gamma(torch.clamp(x, 1e-9, 1e9), gamma)
            elif tf == 'grayscale':
                x = grayscale(x)
            elif tf == 'boxblur':
                x = self.apply_batch(
                    x, boxblur(x), params['boxblur']['p'], num_total)
            elif tf == 'medblur':
                x = self.apply_batch(
                    x, medblur(x), params['medblur']['p'], num_total)
            elif tf == 'gaussblur':
                x = self.apply_batch(
                    x, gaussblur(x), params['gaussblur']['p'], num_total)
            elif tf == 'motionblur':
                x = self.apply_batch(
                    x, motionblur(x), params['motionblur']['p'], num_total)
            elif tf == 'laplacian':
                x = self.apply_batch(
                    x, laplacian(x), params['laplacian']['p'], num_total)
            elif tf == 'sobel':
                x = self.apply_batch(
                    x, sobel(x), params['sobel']['p'], num_total)
            # elif tf == 'motionblur':
            #     for i, x_cur in enumerate(x):
            #         x[i] = motionblur(x_cur)
            else:
                print(tf)
                raise NotImplementedError(
                    'Specified transformation is not implemented.')

        # apply affine transformation
        grid = F.affine_grid(theta[:, :2, :], x.size())
        x = F.grid_sample(x, grid)

        if params['clip'] is not None:
            x = torch.clamp(x, params['clip'][0], params['clip'][1])

        outputs = self.basic_net(x)
        if num_draws > 1:
            outputs = outputs.view(num_draws, batch_size, -1).permute(1, 0, 2)
        if return_img:
            return outputs, x
        return outputs

    def apply_batch(self, x, temp, p, num_total):
        mask = torch.zeros(num_total, device=x.device).bernoulli_(p).view(
            num_total, 1, 1, 1)
        return x * mask + temp * (1 - mask)

    def classify(self, logits):
        if logits.dim() == 3:
            y_pred = logits.argmax(2).cpu()
            num_classes = logits.size(-1)
            y_pred = np.apply_along_axis(
                lambda z, n=num_classes: np.bincount(z, minlength=n),
                axis=1, arr=y_pred) / float(y_pred.size(1))
            return torch.tensor(y_pred)
        if logits.dim() == 2:
            return F.softmax(logits, 1).cpu()
        raise AssertionError('Wrong logits dimension: %d' % logits.dim())
