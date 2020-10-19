import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import conv2d
from .conv2d_repeat import Conv2dRepeat

cfg = {
    'VGG10' : [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG9' : [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 'M'],
    'VGG8' : [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M'],
    'VGG7' : [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 'M'],
    'VGG6' : [64, 64, 'M', 128, 128, 'M', 256, 256, 'M'],
    'VGG5' : [64, 64, 'M', 128, 128, 'M', 256, 'M'],
    'VGG4' : [64, 64, 'M', 128, 128, 'M'],
    'VGG3' : [64, 64, 'M', 128, 'M'],
}

class VGG(nn.Module):
    def __init__(self, vgg_name, num_classes):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.vgg_name = vgg_name
        if vgg_name not in ['VGG3','VGG5','VGG4','VGG6']:
            self.classifier = nn.Linear(512, num_classes)
        elif vgg_name in ['VGG5', 'VGG6']:
            self.classifier = nn.Linear(256, num_classes)
        else:
            self.classifier = nn.Linear(128, num_classes)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        out = self.features(x)
        if self.vgg_name in ['VGG3','VGG4','VGG5','VGG6','VGG7','VGG8']:
            out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
    
class CVGG13_3(nn.Module):
    def __init__(self, num_classes=10):
        super(CVGG13_3, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = Conv2dRepeat((128,64,3,3), (128,128,3,3), stride = 1, padding = 1, conv_type="inter", weight_activation = "swish")
        self.conv3_1 = Conv2dRepeat((128,64,3,3), (256,128,3,3), stride = 1, padding = 1, conv_type="inter", weight_activation = "swish")
        self.conv3_2 = Conv2dRepeat((128,64,3,3), (256,256,3,3), stride = 1, padding = 1, conv_type="inter", weight_activation = "swish")
        self.conv4_1 = Conv2dRepeat((128,64,3,3), (512,256,3,3), stride = 1, padding = 1, conv_type="inter", weight_activation = "swish")
        self.conv4_2 = Conv2dRepeat((128,64,3,3), (512,512,3,3), stride = 1, padding = 1, conv_type="inter", weight_activation = "swish")
        self.conv5_1 = Conv2dRepeat((128,64,3,3), (512,512,3,3), stride = 1, padding = 1, conv_type="inter", weight_activation = "swish")
        self.conv5_2 = Conv2dRepeat((128,64,3,3), (512,512,3,3), stride = 1, padding = 1, conv_type="inter", weight_activation = "swish")
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(256)
        self.bn6 = nn.BatchNorm2d(256)
        self.bn7 = nn.BatchNorm2d(512)
        self.bn8 = nn.BatchNorm2d(512)
        self.bn9 = nn.BatchNorm2d(512)
        self.bn10 = nn.BatchNorm2d(512)
        self.fc = nn.Linear(512, num_classes)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avgpool = nn.AvgPool2d(kernel_size=1, stride=1)

    def forward(self, x):
        x = self.conv1_1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv1_2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.maxpool(x)
        
        x = self.conv2_1(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.conv2_2(x, self.conv2_1.weight)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.maxpool(x)
        
        x = self.conv3_1(x, self.conv2_1.weight)
        x = self.bn5(x)
        x = F.relu(x)
        x = self.conv3_2(x, self.conv2_1.weight)
        x = self.bn6(x)
        x = F.relu(x)
        x = self.maxpool(x)
        
        x = self.conv4_1(x, self.conv2_1.weight)
        x = self.bn7(x)
        x = F.relu(x)
        x = self.conv4_2(x, self.conv2_1.weight)
        x = self.bn8(x)
        x = F.relu(x)
        x = self.maxpool(x)
        
        x = self.conv5_1(x, self.conv2_1.weight)
        x = self.bn9(x)
        x = F.relu(x)
        x = self.conv5_2(x, self.conv2_1.weight)
        x = self.bn10(x)
        x = F.relu(x)
        x = self.maxpool(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
class CVGG13_4(nn.Module):
    def __init__(self, num_classes=10):
        super(CVGG13_4, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3_1 = Conv2dRepeat((128,64,3,3), (256,128,3,3), stride = 1, padding = 1, conv_type="inter", weight_activation = "swish")
        self.conv3_2 = Conv2dRepeat((128,128,3,3), (256,256,3,3), stride = 1, padding = 1, conv_type="inter", weight_activation = "swish")
        self.conv4_1 = Conv2dRepeat((128,64,3,3), (512,256,3,3), stride = 1, padding = 1, conv_type="inter", weight_activation = "swish")
        self.conv4_2 = Conv2dRepeat((128,128,3,3), (512,512,3,3), stride = 1, padding = 1, conv_type="inter", weight_activation = "swish")
        self.conv5_1 = Conv2dRepeat((128,64,3,3), (512,512,3,3), stride = 1, padding = 1, conv_type="inter", weight_activation = "swish")
        self.conv5_2 = Conv2dRepeat((128,128,3,3), (512,512,3,3), stride = 1, padding = 1, conv_type="inter", weight_activation = "swish")
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(256)
        self.bn6 = nn.BatchNorm2d(256)
        self.bn7 = nn.BatchNorm2d(512)
        self.bn8 = nn.BatchNorm2d(512)
        self.bn9 = nn.BatchNorm2d(512)
        self.bn10 = nn.BatchNorm2d(512)
        self.fc = nn.Linear(512, num_classes)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avgpool = nn.AvgPool2d(kernel_size=1, stride=1)

    def forward(self, x):
        x = self.conv1_1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv1_2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.maxpool(x)
        
        x = self.conv2_1(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.conv2_2(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.maxpool(x)
        
        x = self.conv3_1(x, self.conv2_1.weight)
        x = self.bn5(x)
        x = F.relu(x)
        x = self.conv3_2(x, self.conv2_2.weight)
        x = self.bn6(x)
        x = F.relu(x)
        x = self.maxpool(x)
        
        x = self.conv4_1(x, self.conv2_1.weight)
        x = self.bn7(x)
        x = F.relu(x)
        x = self.conv4_2(x, self.conv2_2.weight)
        x = self.bn8(x)
        x = F.relu(x)
        x = self.maxpool(x)
        
        x = self.conv5_1(x, self.conv2_1.weight)
        x = self.bn9(x)
        x = F.relu(x)
        x = self.conv5_2(x, self.conv2_2.weight)
        x = self.bn10(x)
        x = F.relu(x)
        x = self.maxpool(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
class CVGG13_5(nn.Module):
    def __init__(self, num_classes=10):
        super(CVGG13_5, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = Conv2dRepeat((128,64,3,3), (256,256,3,3), stride = 1, padding = 1, conv_type="inter", weight_activation = "swish")
        self.conv4_1 = Conv2dRepeat((128,128,3,3), (512,256,3,3), stride = 1, padding = 1, conv_type="inter", weight_activation = "swish")
        self.conv4_2 = Conv2dRepeat((256,128,3,3), (512,512,3,3), stride = 1, padding = 1, conv_type="inter", weight_activation = "swish")
        self.conv5_1 = Conv2dRepeat((128,128,3,3), (512,512,3,3), stride = 1, padding = 1, conv_type="inter", weight_activation = "swish")
        self.conv5_2 = Conv2dRepeat((256,128,3,3), (512,512,3,3), stride = 1, padding = 1, conv_type="inter", weight_activation = "swish")
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(256)
        self.bn6 = nn.BatchNorm2d(256)
        self.bn7 = nn.BatchNorm2d(512)
        self.bn8 = nn.BatchNorm2d(512)
        self.bn9 = nn.BatchNorm2d(512)
        self.bn10 = nn.BatchNorm2d(512)
        self.fc = nn.Linear(512, num_classes)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avgpool = nn.AvgPool2d(kernel_size=1, stride=1)

    def forward(self, x):
        x = self.conv1_1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv1_2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.maxpool(x)
        
        x = self.conv2_1(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.conv2_2(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.maxpool(x)
        
        x = self.conv3_1(x)
        x = self.bn5(x)
        x = F.relu(x)
        x = self.conv3_2(x, self.conv2_1.weight)
        x = self.bn6(x)
        x = F.relu(x)
        x = self.maxpool(x)
        
        x = self.conv4_1(x, self.conv2_2.weight)
        x = self.bn7(x)
        x = F.relu(x)
        x = self.conv4_2(x, self.conv3_1.weight)
        x = self.bn8(x)
        x = F.relu(x)
        x = self.maxpool(x)
        
        x = self.conv5_1(x, self.conv2_2.weight)
        x = self.bn9(x)
        x = F.relu(x)
        x = self.conv5_2(x, self.conv3_1.weight)
        x = self.bn10(x)
        x = F.relu(x)
        x = self.maxpool(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
class CVGG13_6(nn.Module):
    def __init__(self, num_classes=10):
        super(CVGG13_6, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv4_1 = Conv2dRepeat((128,64,3,3), (512,256,3,3), stride = 1, padding = 1, conv_type="inter", weight_activation = "swish")
        self.conv4_2 = Conv2dRepeat((128,128,3,3), (512,512,3,3), stride = 1, padding = 1, conv_type="inter", weight_activation = "swish")
        self.conv5_1 = Conv2dRepeat((256,128,3,3), (512,512,3,3), stride = 1, padding = 1, conv_type="inter", weight_activation = "swish")
        self.conv5_2 = Conv2dRepeat((256,256,3,3), (512,512,3,3), stride = 1, padding = 1, conv_type="inter", weight_activation = "swish")
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(256)
        self.bn6 = nn.BatchNorm2d(256)
        self.bn7 = nn.BatchNorm2d(512)
        self.bn8 = nn.BatchNorm2d(512)
        self.bn9 = nn.BatchNorm2d(512)
        self.bn10 = nn.BatchNorm2d(512)
        self.fc = nn.Linear(512, num_classes)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avgpool = nn.AvgPool2d(kernel_size=1, stride=1)

    def forward(self, x):
        x = self.conv1_1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv1_2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.maxpool(x)
        
        x = self.conv2_1(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.conv2_2(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.maxpool(x)
        
        x = self.conv3_1(x)
        x = self.bn5(x)
        x = F.relu(x)
        x = self.conv3_2(x)
        x = self.bn6(x)
        x = F.relu(x)
        x = self.maxpool(x)
        
        x = self.conv4_1(x, self.conv2_1.weight)
        x = self.bn7(x)
        x = F.relu(x)
        x = self.conv4_2(x, self.conv2_2.weight)
        x = self.bn8(x)
        x = F.relu(x)
        x = self.maxpool(x)
        
        x = self.conv5_1(x, self.conv3_1.weight)
        x = self.bn9(x)
        x = F.relu(x)
        x = self.conv5_2(x, self.conv3_2.weight)
        x = self.bn10(x)
        x = F.relu(x)
        x = self.maxpool(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
class CVGG13_7(nn.Module):
    def __init__(self, num_classes=10):
        super(CVGG13_7, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = Conv2dRepeat((256,128,3,3), (512,512,3,3), stride = 1, padding = 1, conv_type="inter", weight_activation = "swish")
        self.conv5_1 = Conv2dRepeat((256,256,3,3), (512,512,3,3), stride = 1, padding = 1, conv_type="inter", weight_activation = "swish")
        self.conv5_2 = Conv2dRepeat((512,256,3,3), (512,512,3,3), stride = 1, padding = 1, conv_type="inter", weight_activation = "swish")
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(256)
        self.bn6 = nn.BatchNorm2d(256)
        self.bn7 = nn.BatchNorm2d(512)
        self.bn8 = nn.BatchNorm2d(512)
        self.bn9 = nn.BatchNorm2d(512)
        self.bn10 = nn.BatchNorm2d(512)
        self.fc = nn.Linear(512, num_classes)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avgpool = nn.AvgPool2d(kernel_size=1, stride=1)

    def forward(self, x):
        x = self.conv1_1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv1_2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.maxpool(x)
        
        x = self.conv2_1(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.conv2_2(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.maxpool(x)
        
        x = self.conv3_1(x)
        x = self.bn5(x)
        x = F.relu(x)
        x = self.conv3_2(x)
        x = self.bn6(x)
        x = F.relu(x)
        x = self.maxpool(x)
        
        x = self.conv4_1(x)
        x = self.bn7(x)
        x = F.relu(x)
        x = self.conv4_2(x, self.conv3_1.weight)
        x = self.bn8(x)
        x = F.relu(x)
        x = self.maxpool(x)
        
        x = self.conv5_1(x, self.conv3_2.weight)
        x = self.bn9(x)
        x = F.relu(x)
        x = self.conv5_2(x, self.conv4_1.weight)
        x = self.bn10(x)
        x = F.relu(x)
        x = self.maxpool(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class CVGG13_8(nn.Module):
    def __init__(self, num_classes=10):
        super(CVGG13_8, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_1 = Conv2dRepeat((512,256,3,3), (512,512,3,3), stride = 1, padding = 1, conv_type="inter", weight_activation = "swish")
        self.conv5_2 = Conv2dRepeat((512,512,3,3), (512,512,3,3), stride = 1, padding = 1, conv_type="inter", weight_activation = "swish")
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(256)
        self.bn6 = nn.BatchNorm2d(256)
        self.bn7 = nn.BatchNorm2d(512)
        self.bn8 = nn.BatchNorm2d(512)
        self.bn9 = nn.BatchNorm2d(512)
        self.bn10 = nn.BatchNorm2d(512)
        self.fc = nn.Linear(512, num_classes)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avgpool = nn.AvgPool2d(kernel_size=1, stride=1)

    def forward(self, x):
        x = self.conv1_1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv1_2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.maxpool(x)
        
        x = self.conv2_1(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.conv2_2(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.maxpool(x)
        
        x = self.conv3_1(x)
        x = self.bn5(x)
        x = F.relu(x)
        x = self.conv3_2(x)
        x = self.bn6(x)
        x = F.relu(x)
        x = self.maxpool(x)
        
        x = self.conv4_1(x)
        x = self.bn7(x)
        x = F.relu(x)
        x = self.conv4_2(x)
        x = self.bn8(x)
        x = F.relu(x)
        x = self.maxpool(x)
        
        x = self.conv5_1(x, self.conv4_1.weight)
        x = self.bn9(x)
        x = F.relu(x)
        x = self.conv5_2(x, self.conv4_2.weight)
        x = self.bn10(x)
        x = F.relu(x)
        x = self.maxpool(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
class CVGG13_9(nn.Module):
    def __init__(self, num_classes=10):
        super(CVGG13_9, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = Conv2dRepeat((512,512,3,3), (512,512,3,3), stride = 1, padding = 1, conv_type="inter", weight_activation = "swish")
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(256)
        self.bn6 = nn.BatchNorm2d(256)
        self.bn7 = nn.BatchNorm2d(512)
        self.bn8 = nn.BatchNorm2d(512)
        self.bn9 = nn.BatchNorm2d(512)
        self.bn10 = nn.BatchNorm2d(512)
        self.fc = nn.Linear(512, num_classes)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avgpool = nn.AvgPool2d(kernel_size=1, stride=1)

    def forward(self, x):
        x = self.conv1_1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv1_2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.maxpool(x)
        
        x = self.conv2_1(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.conv2_2(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.maxpool(x)
        
        x = self.conv3_1(x)
        x = self.bn5(x)
        x = F.relu(x)
        x = self.conv3_2(x)
        x = self.bn6(x)
        x = F.relu(x)
        x = self.maxpool(x)
        
        x = self.conv4_1(x)
        x = self.bn7(x)
        x = F.relu(x)
        x = self.conv4_2(x)
        x = self.bn8(x)
        x = F.relu(x)
        x = self.maxpool(x)
        
        x = self.conv5_1(x)
        x = self.bn9(x)
        x = F.relu(x)
        x = self.conv5_2(x, self.conv5_1.weight)
        x = self.bn10(x)
        x = F.relu(x)
        x = self.maxpool(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x