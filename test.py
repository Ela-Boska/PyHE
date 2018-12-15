import vhe
import numpy as np
import layer
import torch

def Dense_test():
    print('running Dense layer test.')
    HE = vhe.HE()
    x = (np.random.randn(1,28*28)*1000).astype(int).astype(object)

    T = HE.TGen(28*28)
    S = HE.getSecretkey(T)

    linear1 = layer.Dense(28*28,100)
    S_new = linear1.build(HE,S)
    x_encrypted = HE.encrypt(T=T,x=x,batching=True)
    y_encrypted = linear1.crypted_forward(x_encrypted,HE)
    yy = HE.decrypt(S=S_new,c=y_encrypted,batching=True)
    y = linear1(torch.from_numpy(x.astype(np.float32)))
    delta_max = (yy-y).abs().max()
    print('max value difference is {}'.format(delta_max))

def Conv_test():
    print('running Conv layer test.')
    HE = vhe.HE()
    x = (np.random.randn(1,1,28*28)*1000).astype(int).astype(object)

    T = HE.TGen(28*28)
    S = HE.getSecretkey(T)

    conv = layer.Conv(in_channels=1,out_channels=32,kernel_size=3,stride=1,padding=1)
    S_new,_ = conv.build(HE,S,28*28)

    x_encrypted = HE.encrypt(T=T,x=x.reshape(1,-1),batching=True)
    y_encrypted = conv.crypted_forward(x_encrypted,HE)
    yy = HE.decrypt(S=S_new,c=y_encrypted,batching=True)

    y = conv(torch.from_numpy(x.astype(np.float32)))
    delta_max = (yy-y).abs().max()
    print('max value difference is {}'.format(delta_max))