import vhe
import numpy as np
import torch
from torch import nn
from torch.nn import Module,Linear,Parameter
import torch.nn.functional as F
import math
import pdb

a = nn.Linear(10,10)
b = nn.Conv2d
relu_coe = []
nn.ReLU

W = np.zeros(1000,dtype=object).reshape(10,10,10)
for i in range(10):
    W[i,i,i] = 1

def _relu(ones,input,coe,Ms,HE):
    order = 3
    temp = input
    x = np.ones(input.shape,dtype=object)
    ans = ones*coe[0]
    for i in range(order):
        temp = HE.innerProd(input,temp,Ms[i])
        ans += temp*coe[i+1]
    return ans

class Dense(torch.nn.Linear):

    def forward(self, input):
        if input.dim != 2:
            input = input.view(len(input),-1)
        return super().forward(input)

    def crypted_forward(self,input):
        weight = vhe.double(self.weight.detach().numpy())
        ans = weight.dot(input)
        if self.bias_encrypted is not None:
            ans = ans + self.bias_encrypted
        return ans

    def build(self,HE,T):
        bias = self.bias.detach().numpy()
        bias = (1000*bias).astype(int)
        bias = vhe.double(bias)
        batch_size = len(T)
        bias = bias.reshape(-1,1).repeat(1,batch_size)
        self.bias_encrypted = HE.encrypt(T=T,x=bias,batching=True)
        

class Conv(torch.nn.Conv2d):
    
    def crypted_forward(self,input,HE):
        ans = HE.linearTransform(self.M,input,batching=True)
        if self.bias_encrypted is not None:
            ans = ans + self.bias_encrypted
        return ans

    def build(self,HE,S_old,input_size):
        in_channels = self.in_channels
        out_channels = self.out_channels
        H = (input_size/in_channels)**0.5
        assert H==int(H), 'input size = {0} x {1} x {1}'.format(in_channels,H)
        H = int(H)
        #pdb.set_trace()
        HH = (self.padding[0]*2+H-self.kernel_size[0])//self.stride[0] + 1
        output_size = out_channels*HH**2
        T_new = HE.TGen(output_size)
        _weight = self.weight.detach().numpy()
        _bias = self.bias.detach().numpy()
        _bias = _bias.repeat(HH*HH)
        self.M = np.zeros([out_channels,HH,HH,in_channels,H+2*self.padding[0],H+2*self.padding[0]]).astype(object)
        for a in range(out_channels):
            for b in range(HH):
                for c in range(HH):
                    self.M[a,b,c,:,b*self.stride[0]:b*self.stride[0]+self.kernel_size[0],c*self.stride[0]:c*self.stride[0]+self.kernel_size[0]] = _weight[a]
        self.M = self.M[:,:,:,:,self.padding[0]:-self.padding[0],self.padding[0]:-self.padding[0]]
        self.M = self.M.reshape(output_size,input_size)
        self.M = HE.linearTransformClient(self.M,S_old,T_new)
        self.bias_encrypted = HE.encrypt(T=T_new,x=_bias).reshape(1,-1)
        return HE.getSecretkey(T_new),output_size

class poly(Module):

    def __init__(self,coe):
        super().__init__()
        self.coe = coe

    def forward(self, input):
        ans = self.coe[0]
        temp = 1
        for i in range(len(self.coe)-1):
            temp = temp * input
            ans = ans + self.coe[i+1]*temp
        return ans

def relu(w):
    return poly([0.25*w,0.5,0.25/w])