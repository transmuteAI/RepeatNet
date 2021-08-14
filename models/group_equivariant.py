# from e2cnn import gspaces
import numpy as np
import math
# from e2cnn import nn as enn
import torch
from torch.nn import Conv2d, BatchNorm2d, MaxPool2d, AdaptiveAvgPool2d, ELU
from .conv2d_repeat import *

# class SteerableCNN(torch.nn.Module):
    
#     def __init__(self, n_classes=10, N=16, k=0.5):
        
#         super(SteerableCNN, self).__init__()
        
#         self.r2_act = gspaces.Rot2dOnR2(N=N)
#         # the input image is a 1-channel field, that doesn't transform when input is rotated or flipped
#         in_type = enn.FieldType(self.r2_act, 1*[self.r2_act.trivial_repr])
        
#         # we store the input type for wrapping the images into a geometric tensor during the forward pass
#         self.input_type = in_type
        
#         # convolution 1
#         # first specify the output type of the convolutional layer
#         out_type = enn.FieldType(self.r2_act, round(k*16)*[self.r2_act.regular_repr])
#         self.block1 = enn.SequentialModule(
#             enn.R2Conv(in_type, out_type, kernel_size=7, padding=1, bias=False, 
#                        padding_mode = 'constant', sigma = [0.6, 0.6, 0.6, 0.4], 
#                        frequencies_cutoff=lambda r: 0 if r==0 else (3 if r==2 else 2), rings = [0,1,2,3]),
#             enn.InnerBatchNorm(out_type),
#             enn.ELU(out_type, inplace=True)
#         )
        
#         # convolution 2
#         # the old output type is the input type to the next layer
#         in_type = self.block1.out_type
#         out_type = enn.FieldType(self.r2_act, round(k*24)*[self.r2_act.regular_repr])
#         self.block2 = enn.SequentialModule(
#             enn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False, 
#                        padding_mode = 'constant', sigma = [0.6, 0.6, 0.4], 
#                        frequencies_cutoff=lambda r: 0 if r==0 else 2, rings = [0,1,2]),
#             enn.InnerBatchNorm(out_type),
#             enn.ELU(out_type, inplace=True)
#         )
#         self.pool1 = enn.SequentialModule(
#             enn.PointwiseMaxPool(out_type, kernel_size  = 2, stride=2)
#         )
        
#         # convolution 3
#         # the old output type is the input type to the next layer
#         in_type = self.block2.out_type
#         out_type = enn.FieldType(self.r2_act, round(k*32)*[self.r2_act.regular_repr])
#         self.block3 = enn.SequentialModule(
#             enn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False, 
#                        padding_mode = 'constant', sigma = [0.6, 0.6, 0.4], 
#                        frequencies_cutoff=lambda r: 0 if r==0 else 2, rings = [0,1,2]),
#             enn.InnerBatchNorm(out_type),
#             enn.ELU(out_type, inplace=True)
#         )
        
#         # convolution 4
#         # the old output type is the input type to the next layer
#         in_type = self.block3.out_type
#         out_type = enn.FieldType(self.r2_act, round(k*32)*[self.r2_act.regular_repr])
#         self.block4 = enn.SequentialModule(
#             enn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False, 
#                        padding_mode = 'constant', sigma = [0.6, 0.6, 0.4], 
#                        frequencies_cutoff=lambda r: 0 if r==0 else 2, rings = [0,1,2]),
#             enn.InnerBatchNorm(out_type),
#             enn.ELU(out_type, inplace=True)
#         )
#         self.pool2 = enn.SequentialModule(
#             enn.PointwiseMaxPool(out_type, kernel_size  = 2, stride=2)
#         )
        
#         # convolution 5
#         # the old output type is the input type to the next layer
#         in_type = self.block4.out_type
#         out_type = enn.FieldType(self.r2_act, round(k*48)*[self.r2_act.regular_repr])
#         self.block5 = enn.SequentialModule(
#             enn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False, 
#                        padding_mode = 'constant', sigma = [0.6, 0.6, 0.4], 
#                        frequencies_cutoff=lambda r: 0 if r==0 else 2, rings = [0,1,2]),
#             enn.InnerBatchNorm(out_type),
#             enn.ELU(out_type, inplace=True)
#         )
        
#         # convolution 6
#         # the old output type is the input type to the next layer
#         in_type = self.block5.out_type
#         out_type = enn.FieldType(self.r2_act, round(k*64)*[self.r2_act.regular_repr])
#         self.block6 = enn.SequentialModule(
#             enn.R2Conv(in_type, out_type, kernel_size=5, padding=0, bias=False, 
#                        padding_mode = 'constant', sigma = [0.6, 0.6, 0.4], 
#                        frequencies_cutoff=lambda r: 0 if r==0 else 2, rings = [0,1,2]),
#             enn.InnerBatchNorm(out_type),
#             enn.ELU(out_type, inplace=True)
#         )
#         self.pool3 = enn.PointwiseAdaptiveAvgPool(out_type, 1)
        
#         self.gpool = enn.GroupPooling(out_type)

#         c = self.gpool.out_type.size

#         # Fully Connected
#         self.fully_net = torch.nn.Sequential(
#             torch.nn.Linear(c, 64),
#             torch.nn.BatchNorm1d(64),
#             torch.nn.ELU(inplace=True),
#             torch.nn.Linear(64, n_classes),
#         )
        
    
#     def forward(self, input: torch.Tensor):
#         # wrap the input tensor in a GeometricTensor
#         # (associate it with the input type)
#         x = enn.GeometricTensor(input, self.input_type)
        
#         x = self.block1(x)
#         x = self.block2(x)
#         x = self.pool1(x)
        
#         x = self.block3(x)
#         x = self.block4(x)
#         x = self.pool2(x)
        
#         x = self.block5(x)
#         x = self.block6(x)

#         # pool over the spatial dimensions
#         x = self.pool3(x)     
        
#         # pool over the group
#         x = self.gpool(x)
#         # unwrap the output GeometricTensor
#         # (take the Pytorch tensor and discard the associated representation)
#         x = x.tensor

#         x = self.fully_net(x.reshape(x.shape[0], -1))
        
#         return x
    
class NormalCNN(torch.nn.Module):
    
    def __init__(self, n_classes=10):
        
        super(NormalCNN, self).__init__()
        k = 1/math.sqrt(2)
        self.conv1 = Conv2d(1, int(32*k), kernel_size=(7, 7), stride=(1, 1), padding=(1, 1), bias=False)
        self.bn1 = BatchNorm2d(int(32*k), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2 = Conv2d(int(32*k), int(48*k), kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=False)
        self.bn2 = BatchNorm2d(int(48*k), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv3 = Conv2d(int(48*k), int(64*k), kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=False)
        self.bn3 = BatchNorm2d(int(64*k), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv4 = Conv2d(int(64*k), int(64*k), kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=False)
        self.bn4 = BatchNorm2d(int(64*k), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv5 = Conv2d(int(64*k), int(96*k), kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=False)
        self.bn5 = BatchNorm2d(int(96*k), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv6 = Conv2d(int(96*k), int(128*k), kernel_size=(5, 5), stride=(1, 1), bias=False)
        self.bn6 = BatchNorm2d(int(128*k), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.mpool = MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0), dilation=(1, 1), ceil_mode=False)
        self.avgpool = AdaptiveAvgPool2d(1)
        self.act = ELU(alpha=1.0)
        # Fully Connected
        self.fully_net = torch.nn.Sequential(
            torch.nn.Linear(int(128*k), 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.ELU(inplace=True),
            torch.nn.Linear(64, n_classes),
        )
        
    
    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        # print(x.shape)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)
        x = self.mpool(x)
        # print(x.shape)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act(x)
        # print(x.shape)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.act(x)
        x = self.mpool(x)
        # print(x.shape)
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.act(x)
        # print(x.shape)
        x = self.conv6(x)
        x = self.bn6(x)
        x = self.act(x)
        # print(x.shape)
        x = self.avgpool(x).squeeze(-1).squeeze(-1)
        # print(x.shape)
        x = self.fully_net(x)
        
        return x
    
class RepCNN(NormalCNN):
    
    def __init__(self, n_classes=10, rep_fact=4*math.sqrt(2), args=None):
        
        super(NormalCNN, self).__init__()
        k = 1/math.sqrt(2)
        self.conv1 = Conv2dRepeat((int(32*k), 1, 7, 7), (int(32*k*rep_fact), 1, 7, 7),stride=1, padding=1, args=args)
        self.bn1 = BatchNorm2d(int(32*k*rep_fact), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2 = Conv2dRepeat((int(48*k), int(32*k), 5, 5), (int(48*k*rep_fact), int(32*k*rep_fact), 5, 5),stride=1, padding=2, args=args)
        self.bn2 = BatchNorm2d(int(48*k*rep_fact), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv3 = Conv2dRepeat((int(64*k), int(48*k), 5, 5), (int(64*k*rep_fact), int(48*k*rep_fact), 5, 5),stride=1, padding=2, args=args)
        self.bn3 = BatchNorm2d(int(64*k*rep_fact), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv4 = Conv2dRepeat((int(64*k), int(64*k), 5, 5), (int(64*k*rep_fact), int(64*k*rep_fact), 5, 5),stride=1, padding=2, args=args)
        self.bn4 = BatchNorm2d(int(64*k*rep_fact), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv5 = Conv2dRepeat((int(96*k), int(64*k), 5, 5), (int(96*k*rep_fact), int(64*k*rep_fact), 5, 5),stride=1, padding=2, args=args)
        self.bn5 = BatchNorm2d(int(96*k*rep_fact), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv6 = Conv2dRepeat((int(128*k), int(96*k), 5, 5), (int(128*k*rep_fact), int(96*k*rep_fact), 5, 5),stride=1, padding=0, args=args)
        self.bn6 = BatchNorm2d(int(128*k*rep_fact), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.mpool = MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0), dilation=(1, 1), ceil_mode=False)
        self.avgpool = AdaptiveAvgPool2d(1)
        self.act = ELU(alpha=1.0)
        # Fully Connected
        self.fully_net = torch.nn.Sequential(
            torch.nn.Linear(int(128*k*rep_fact), 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.ELU(inplace=True),
            torch.nn.Linear(64, n_classes),
        )
    
def c16(num_classes):
    model = SteerableCNN(num_classes, 16, 0.5)
    return model

def c8(num_classes):
    model = SteerableCNN(num_classes, 8, 0.5*math.sqrt(2))
    return model

def c4(num_classes):
    model = SteerableCNN(num_classes, 4, 0.5*2)
    return model

def baseline(num_classes):
    model = NormalCNN(num_classes, )
    return model

def c16_rep(num_classes, args):
    model = RepCNN(num_classes, 4*math.sqrt(2), args)
    return model

def c8_rep(num_classes, args):
    model = RepCNN(num_classes, 4, args)
    return model

def c4_rep(num_classes, args):
    model = RepCNN(num_classes, 2*math.sqrt(2), args)
    return model
