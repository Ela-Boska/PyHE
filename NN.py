import HElib
import Chev
import numpy as np
import torch
from torch import nn
from torch.nn import Module,Linear,Parameter
import torch.nn.functional as F
import math
import pdb

W = np.zeros(1000,dtype=object).reshape(10,10,10)
for i in range(10):
    W[i,i,i] = 1

class Dense(torch.nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        self.bias_encrypted = None
        super().__init__(in_features, out_features, bias=bias)

    def forward(self, input):
        if input.dim != 2:
            input = input.view(len(input),-1)
        return super().forward(input)

    def crypted_forward(self,input):
        if input.dim != 2:
            input = input.view(len(input),-1)
        return F.linear(input,self.weight,self.bias_encrypted)

    def build(self,HE,T,scale,group):
        if self.bias is not None:
            bias = self.bias.type(T.dtype)*scale
            bias = bias.view(1,-1).repeat(T.shape[0],1)
            self.bias_encrypted = HE.encrypt(x=bias,T=T).repeat(group,1)
        

class Conv(torch.nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        self.bias_encrypted = 0
        return super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
    
    def crypted_forward(self,input):
        ans = F.conv2d(input, self.weight, None, self.stride,
                        self.padding, self.dilation, self.groups)
        try:
            ans = ans + self.bias_encrypted
        except:
            pdb.set_trace()
        return ans

    def build(self,HE,T,scale,group):
        if self.bias is not None:
            bias = self.bias.type(T.dtype)*scale
            bias = bias.view(1,-1).repeat(T.shape[0],1)
            self.bias_encrypted = HE.encrypt(x=bias,T=T).view(-1,len(self.bias),1,1).repeat(group,1,1,1)


        
class poly(Module):

    def __init__(self,coe):
        super().__init__()
        self.polynomial = Chev.polynomial(coe)

    def forward(self, input):
        ans = self.coe[0]
        temp = 1
        for i in range(len(self.coe)-1):
            temp = temp * input
            ans = ans + self.coe[i+1]*temp
        return ans

class polynomial(Module):
    
    def __init__(self, coe):
        self.coe = torch.tensor(coe)
        self.degree = len(coe)-1
        super().__init__()

    def cuda(self,device=None):
        self.coe = self.coe.cuda(device)
        return super().cuda(device)

    def double(self):
        self.coe = self.coe.double()
        return super().double()

    def float(self):
        self.coe = self.coe.float()
        return super().float()

    def forward(self,x):
        dim = x.dim()
        x = x.view(1,*x.shape)
        shape = [-1]+[1]*dim
        ans = x**torch.arange(self.degree+1,dtype=x.dtype,device=x.device).view(shape)
        ans = ans * self.coe.view(*shape)
        return torch.sum(ans,0)
    
    def __repr__(self):
        return "ploynomial layer with coefficients, {}".format(self.coe.tolist())

    def build(self,HE,T,scale,group):
        self.group = group
        self.m = len(T)
        batch_size = len(T)
        W = torch.zeros(batch_size,batch_size,batch_size,dtype=T.dtype,device=T.device)
        for i in range(batch_size):
            W[i,i,i] = 1
        self.M = HE.innerProdClient(W,T)
        self.coe = self.coe.type(T.dtype).to(T.device)
        self.scale_coe = scale*self.coe/scale**torch.arange(len(self.coe),dtype=T.dtype,device=T.device)
        ones = torch.ones(batch_size,dtype=T.dtype,device=T.device)
        self.ones = HE.encrypt(x=ones,T=T)

    def crypted_forward(self,HE,input):
        dim = input.dim()
        shape = [-1,1,1]
        input_shape = input.shape
        input = input.view(self.m+1,-1)
        ans = torch.empty(size=[self.degree+1,*input.shape],dtype=input.dtype,device=input.device)
        ans[0] = self.ones.view(self.m+1,1)
        ans[1] = input
        for i in range(1,self.degree):
            ans[i+1] = HE.innerProd(ans[i],input,self.M)
        ans = ans*self.scale_coe.view(shape)
        return torch.sum(ans,0).view(input_shape)

class AlanNet(Module):

    def __init__(self,l=1):
        super().__init__()
        self.relu = polynomial([l/4,1/2,1/(4*l)])
        self.layers = nn.Sequential(
            Conv(in_channels=1,out_channels=32,kernel_size=5,stride=1,padding=2),
            self.relu,
            nn.AvgPool2d(kernel_size=2,stride=2),
            Conv(in_channels=32,out_channels=64,kernel_size=5,stride=1,padding=2),
            self.relu,
            nn.AvgPool2d(kernel_size=2,stride=2),
            Dense(7*7*64,1024),
            self.relu,
            Dense(1024,10)
        )
        self.lr = []
        self.epoch = 0
        self.precision = [0]

    def forward(self, input):
        for layer in self.layers:
            input = layer(input)
        return input

    def build(self,HE,T,scale,group=1):
        for layer in self.layers:
            if isinstance(layer,Conv) or isinstance(layer,polynomial) or isinstance(layer,Dense):
                layer.build(HE,T,scale,group)
    
    def crypted_forward(self,input,HE):
        for layer in self.layers:
            if isinstance(layer,Conv) or isinstance(layer,Dense):
                input = layer.crypted_forward(input)
            elif isinstance(layer,polynomial):
                input = layer.crypted_forward(HE,input)
            else:
                input = layer(input)
        return input

    def cuda(self,device=None):
        self.relu.cuda()
        return super().cuda(device)

    def double(self):
        self.relu.double()
        return super().double()

    def float(self):
        self.relu.float()
        return super().float()