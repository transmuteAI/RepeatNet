import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import conv2d
import math

class HardBinaryConv(nn.Module):
    def __init__(self, roc, ric, rk1, rk2):
        super(HardBinaryConv, self).__init__()
        self.weight = nn.Parameter(torch.rand(roc, ric, rk1, rk2)*0.001, requires_grad=True)

    def forward(self, x):
        real_weight = self.weight
        binary_weights_no_grad = torch.sign(real_weight)
        cliped_weights = torch.clamp(real_weight, -1.0, 1.0)
        binary_weights = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
        return binary_weights.reshape_as(x)*x

class Conv2dRepeat(nn.Module):
    def __init__(self, original_weight_shape, repeated_weight_shape, previous_weight_shape=None, concat_dim=1, stride=1, padding=1, conv_type="intra", args=None):
        super(Conv2dRepeat, self).__init__()
        self.args = args
        self.ooc, self.oic, self.ok1, self.ok2 = original_weight_shape
        self.roc, self.ric, self.rk1, self.rk2 = repeated_weight_shape
        self.stride = stride
        self.padding = padding
        self.conv_type = conv_type
        self.wactivation = self.args.weight_activation
        self.do_repeat = False if (original_weight_shape==repeated_weight_shape or self.wactivation=='random_bireal') else True
        if previous_weight_shape is not None:
            if concat_dim==0:
                self.ooc+=previous_weight_shape[0]
            else:
                self.oic+=previous_weight_shape[1]
        
        self.r0 = self.roc//self.ooc
        self.r1 = self.ric//self.oic
        
        self.e0 = self.roc%self.ooc
        self.e1 = self.ric%self.oic
        
        self.alphas = None
        self.betas = None
        
        self.bias = nn.Parameter(torch.zeros(self.roc))
        if self.wactivation=='swish' and self.do_repeat:
            self.alphas = nn.Parameter(torch.zeros((1, self.r0*self.r1)))
            self.betas = nn.Parameter(torch.zeros((1, self.r0*self.r1)))
            torch.nn.init.xavier_uniform_(self.alphas)
            torch.nn.init.xavier_uniform_(self.betas)
            if self.e0!=0:
                self.ealpha1 = nn.Parameter(torch.rand((1)))
                self.ebeta1 = nn.Parameter(torch.rand((1)))
            if self.e1!=0:
                self.ealpha2 = nn.Parameter(torch.rand((1, 1)))
                self.ebeta2 = nn.Parameter(torch.rand((1, 1)))

        elif self.wactivation=='static_drop':
            self.drop_mask = torch.ones(self.roc, self.ric, self.rk1, self.rk2)*(1-self.args.drop_rate)
            self.drop_mask = torch.bernoulli(self.drop_mask)
            self.drop_mask = nn.Parameter(self.drop_mask, requires_grad=False)
        
        elif self.wactivation=='bireal':
            self.binary_activation = HardBinaryConv(self.ooc*self.r0, self.oic*self.r1, self.rk1, self.rk2)
            if self.e0!=0:
                self.binary_activation0 = HardBinaryConv(self.e0, self.oic*self.r1, self.rk1, self.rk2)
            if self.e1!=0:
                self.binary_activation1 = HardBinaryConv(self.ooc*self.r0 + self.e0, self.e1, self.rk1, self.rk2)
            
        elif self.wactivation=='random_bireal':
            self.binary_activation = HardBinaryConv(self.roc, self.ric, self.rk1, self.rk2)
            self.random_weights = torch.empty(((self.roc*self.ric)-(self.ooc*self.oic), self.ok1, self.ok2))
            nn.init.kaiming_uniform_(self.random_weights, a=math.sqrt(5))
            self.random_weights = nn.Parameter(self.random_weights, requires_grad=False)

        if self.conv_type!="inter":
            self.weight = nn.Parameter(torch.zeros(original_weight_shape))
            torch.nn.init.xavier_uniform_(self.weight)
            
        if self.conv_type=='hybrid':
            self.dim = concat_dim  
    
    def forward(self, x, weights=None):
        if self.conv_type=="intra":
            weights = self.repeat(self.weight)
        elif self.conv_type=="inter":
            weights = self.repeat(weights)
        else:
            weights = torch.cat((self.weight, weights), dim = self.dim)
            weights = self.repeat(weights)
        if self.e0!=0:
            if self.wactivation=='swish':
                weights = torch.cat((weights, self.activation(weights[:self.e0,...], self.ealpha1, self.ebeta1)), dim=0)
            elif self.wactivation=='bireal':
                weights = torch.cat((weights, self.activation(weights[:self.e0,...], e=0)), dim=0)
        if self.e1!=0:
            if self.wactivation=='swish':
                weights = torch.cat((weights, self.activation(weights[:,:self.e1,...], self.ealpha2, self.ebeta2)), dim=1)
            elif self.wactivation=='bireal':
                weights = torch.cat((weights, self.activation(weights[:,:self.e1,...], e=1)), dim=1)
        x = F.conv2d(x, weights, self.bias, stride=self.stride, padding = self.padding)
        return x
    
    def activation(self, weight, alphas=None, betas=None, e=-1):
        if self.wactivation=="swish":
            alphas = alphas.reshape(1,1,self.r0, self.r1)
            alphas = torch.nn.functional.interpolate(alphas, None, (self.roc//self.r0,self.ric//self.r1), 'nearest')
            alphas = alphas.reshape(self.roc, self.ric,1,1)
            betas = betas.reshape(1,1,self.r0, self.r1)
            betas = torch.nn.functional.interpolate(betas, None, (self.roc//self.r0,self.ric//self.r1), 'nearest')
            betas = betas.reshape(self.roc, self.ric,1,1)
            x = weight*alphas/(1+torch.exp(weight*betas))
        elif self.wactivation=="static_drop":
            x = weight*(self.drop_mask.reshape_as(weight).detach())
        elif self.wactivation=='bireal':
            if e==-1:
                x = self.binary_activation(weight)
            elif e==0:
                x = self.binary_activation0(weight)
            else:
                x = self.binary_activation1(weight)
        elif self.wactivation=='linear':
            x = weight
        elif self.wactivation=='random_bireal':
            x = self.binary_activation(weight)
        return x

    def repeat(self, weights):
        if self.do_repeat:
            weights = weights.unsqueeze(0).expand(self.r0, self.ooc, self.oic, self.ok1, self.ok2).reshape(self.r0*self.ooc, self.oic, self.ok1, self.ok2)
            weights = weights.unsqueeze(1).expand(self.r0*self.ooc, self.r1, self.oic, self.ok1, self.ok2).reshape(self.r0*self.ooc, self.r1*self.oic, self.ok1, self.ok2)
#             weights = weights.repeat((self.r0, self.r1,1,1)) #<- Slow (https://github.com/pytorch/pytorch/issues/43192)
            weights = self.activation(weights,self.alphas,self.betas)
        if self.wactivation=='random_bireal':
            weights = weights.reshape(-1,self.ok1, self.ok2)
            weights = torch.cat([weights, self.random_weights])
            weights = self.activation(weights)
        weights = weights.reshape(self.r0*self.ooc, self.r1*self.oic, self.ok1, self.ok2)
        return weights
