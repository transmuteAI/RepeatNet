import torch.nn as nn
import torch
from .conv2d_repeat import Conv2dRepeat

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
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

class RepeatBlock(nn.Module):
    def __init__(self, inplanes, planes, input_weight_shape, stride=1, downsample=None):
        super(RepeatBlock, self).__init__()
        self.conv1 = Conv2dRepeat(input_weight_shape[0], (inplanes, planes, 3, 3), stride=stride, padding=1, conv_type="inter", weight_activation='swish')
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = Conv2dRepeat(input_weight_shape[1], (planes, planes, 3, 3), stride=1, padding=1, conv_type="inter", weight_activation='swish')
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, input_weight):
        residual = x

        out = self.conv1(x, input_weight[0])
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out, input_weight[1])
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
    
class ResNetCifar(nn.Module):
    def __init__(self, block, rep_block, layers, num_classes=10):
        self.inplanes = 16
        super(ResNetCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1, layer1_shape = self._make_layer(block, 16, layers[0], stride=1)
        self.rep_layer1 = self._make_repeat_layer(layer1_shape, rep_block, 16, layers[0])       
        self.layer2, layer2_shape = self._make_layer(block, 32, layers[1], stride=2)
        self.rep_layer2 = self._make_repeat_layer(layer2_shape, rep_block, 32, layers[1])       
        self.layer3, layer3_shape = self._make_layer(block, 64, layers[2], stride=2)
        self.rep_layer3 = self._make_repeat_layer(layer3_shape, rep_block, 64, layers[2])       
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = []
        layer_shape = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        layer_shape.append([(planes, self.inplanes, 3, 3) , (planes, planes, 3, 3)])
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
            layer_shape.append([(planes, self.inplanes, 3, 3) , (planes, planes, 3, 3)])

        return nn.Sequential(*layers), layer_shape
    
    def _make_repeat_layer(self, prev_layer_shape, block, planes, blocks):        
        layers = []
        for i in range(blocks):
            layers.append(block(planes, planes, prev_layer_shape[i]))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        for Rlayer, OGlayer in zip(self.rep_layer1, self.layer1):
            x = Rlayer(x, [OGlayer.conv1.weight, OGlayer.conv2.weight])
        x = self.layer2(x)
        for Rlayer, OGlayer in zip(self.rep_layer2, self.layer2):
            x = Rlayer(x, [OGlayer.conv1.weight, OGlayer.conv2.weight])
        x = self.layer3(x)
        for Rlayer, OGlayer in zip(self.rep_layer3, self.layer3):
            x = Rlayer(x, [OGlayer.conv1.weight, OGlayer.conv2.weight])
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

def resnet20_38(num_classes=10):
    return ResNetCifar(BasicBlock, RepeatBlock, [3, 3, 3], num_classes)


def resnet32_62(num_classes=10):
    return ResNetCifar(BasicBlock, RepeatBlock, [5, 5, 5], num_classes)


def resnet44_86(num_classes=10):
    return ResNetCifar(BasicBlock, RepeatBlock, [7, 7, 7], num_classes)


def resnet56_110(num_classes=10):
    return ResNetCifar(BasicBlock, RepeatBlock, [9, 9, 9], num_classes)


def resnet110_218(num_classes=10):
    return ResNetCifar(BasicBlock, RepeatBlock, [18, 18, 18], num_classes)


def resnet1202_2402(num_classes=10):
    return ResNetCifar(BasicBlock, RepeatBlock, [200, 200, 200], num_classes)