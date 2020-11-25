import torch.nn as nn
import torch
from .conv2d_repeat import Conv2dRepeat
import math

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    width=1
    def __init__(self, inplanes, planes, stride=1, downsample=None, factor_in=1, factor_out=1, kernel_size=5, padding = 2):
        super(BasicBlock, self).__init__()
        
        self.conv1 = Conv2dRepeat((planes, inplanes, kernel_size, kernel_size), (round(planes*factor_out), round(inplanes*factor_in), kernel_size, kernel_size), stride=stride, padding=padding)
        
        self.bn1 = nn.BatchNorm2d(round(planes*factor_out))
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = Conv2dRepeat((planes, planes, kernel_size, kernel_size), (round(planes*factor_out), round(planes*factor_out), kernel_size, kernel_size), padding=padding)
        self.bn2 = nn.BatchNorm2d(round(planes*factor_out))
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNetCifar(nn.Module):
    def __init__(self, block, layers, factors, width=1, num_classes=10):
        self.inplanes = 16
        super(ResNetCifar, self).__init__()
        self.conv1 = Conv2dRepeat((16, 3, 5, 5), (round(16*factors[0]), 3, 5, 5), stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(round(16*factors[0]))
        self.relu = nn.ReLU(inplace=True)
        block.width = width
        self.layer1 = self._make_layer(block, 16*width, layers[0], stride=1, factor_in = factors[0], factor_out = factors[1], kernel_size = 5, padding=2)
        self.layer2 = self._make_layer(block, 32*width, layers[1], stride=2, factor_in = factors[1], factor_out = factors[2], kernel_size = 3, padding=1)
        self.layer3 = self._make_layer(block, 64*width, layers[2], stride=2, factor_in = factors[2], factor_out = factors[3], kernel_size = 3, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(round(64*width*factors[3]), num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0.01)

    def _make_layer(self, block, planes, blocks, stride=1, factor_in = 1, factor_out = 1, kernel_size = 5, padding = 2):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                Conv2dRepeat((planes, self.inplanes, 1, 1), (round(planes*factor_out), round(self.inplanes*factor_in), 1, 1), stride=stride, padding = 0),
#                 nn.Conv2d(self.inplanes, planes,
#                           kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(round(planes*factor_out)),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, factor_in, factor_out, kernel_size = kernel_size, padding = padding))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, factor_in=factor_out, factor_out=factor_out, kernel_size = kernel_size, padding = padding))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x).squeeze()
        x = self.fc(x)

        return x

def wrn_16_8(num_classes=10, factors = [1,1,1,1]):
    model = ResNetCifar(BasicBlock, [2, 2, 2], factors, width=8, num_classes=num_classes)
    return model

def wrn_28_10_d8d4d1(num_classes=10, factors = [4,4,2*math.sqrt(2),2*math.sqrt(2)]):
    model = ResNetCifar(BasicBlock, [4, 4, 4], factors, width=10, num_classes=num_classes)
    return model