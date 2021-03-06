import torch.nn as nn
import torch
from .conv2d_repeat import Conv2dRepeat

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    width=1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        if inplanes==16:
            self.conv1 = Conv2dRepeat((planes//self.width, inplanes, 3, 3), (planes, inplanes, 3, 3), stride=stride)
        else:
            self.conv1 = Conv2dRepeat((planes//self.width, inplanes//self.width, 3, 3), (planes, inplanes, 3, 3), stride=stride)
        
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = Conv2dRepeat((planes//self.width, planes//self.width, 3, 3), (planes, planes, 3, 3))
        self.bn2 = nn.BatchNorm2d(planes)
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
    def __init__(self, block, layers, width=1, num_classes=10):
        self.inplanes = 16
        super(ResNetCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        block.width = width
        self.width = width
        self.layer1 = self._make_layer(block, 16*width, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 32*width, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64*width, layers[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64*width, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0.01)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1:
            downsample = nn.Sequential(
                Conv2dRepeat((planes//self.width, self.inplanes//self.width, 1, 1), (planes, self.inplanes, 1, 1), stride=stride, padding=0),
                nn.BatchNorm2d(planes),
            )
        elif self.inplanes != planes:
            downsample = nn.Sequential(
                Conv2dRepeat((planes//self.width, self.inplanes, 1, 1), (planes, self.inplanes, 1, 1), stride=stride, padding=0),
                nn.BatchNorm2d(planes),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

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

def resnet_rep(num_classes=10, k=1):
    model = ResNetCifar(BasicBlock, [2, 2, 2], width=k, num_classes=num_classes)
    return model

