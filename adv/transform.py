import torch
import torch.nn as nn
import torch.nn.functional as F


def quantize_tensor(x, num_bits=8):
    qmax = 2.**num_bits - 1.
    # min_val, max_val = 0., 1.
    q_x = torch.round(x * qmax)
    q_x = q_x / qmax
    return q_x


def sigmoid(x, a=1, b=0):
    return 1 / (1 + torch.exp(-a * (x - b)))


def quantize_tensor_soft(x, num_bits=8, a=20):
    qmax = 2.**num_bits - 1.
    step_size = 1 / (qmax + 1)
    q_x = 0
    for i in range(int(qmax)):
        step = step_size * (i + 1)
        q_x += sigmoid(x, a=a, b=step)
    return q_x / qmax


class QuantizeLayer(nn.Module):
    def __init__(self, num_bits, soft):
        super(QuantizeLayer, self).__init__()
        self.num_bits = num_bits
        self.soft = soft

    def forward(self, input):
        if self.soft:
            return quantize_tensor_soft(input, num_bits=self.num_bits, a=20)
        return quantize_tensor(input, num_bits=self.num_bits)


class PreActResNetQuant(nn.Module):
    def __init__(self, block, num_blocks, num_bits=8, soft=False, num_classes=10):
        super(PreActResNetQuant, self).__init__()
        self.in_planes = 64

        # self.mean = nn.Parameter(
        #     data=torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1),
        #     requires_grad=False)
        # self.std = nn.Parameter(
        #     data=torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1),
        #     requires_grad=False)

        self.quant = QuantizeLayer(num_bits, soft)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # x = (x - self.mean) / self.std
        out = self.quant(x)
        out = self.conv1(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
