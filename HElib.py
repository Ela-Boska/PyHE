import torch
import numpy as np
import decimal

double = np.vectorize(decimal.Decimal)
floor = np.vectorize(int)
import pdb

r"""
The followings are the error caused by operations:
    encrypt:            eBound
    scale:              1/6
    linear transform    eBound
    add                 eBound
    inner product       m^(1/2) * tBound /6
So for a network with 20 layers, batch size(m) is 10, the error caused is
about 20*eBound*2(linear transform + bias) + (10)^0.5*tBound/6*20 = 5e4=2^16, so to be save, let's make w be 2^3 times as large as 2^16, 2^19. 
Note that the error caused by the network is proportional to the depth of layers and the sqaure root of 
the size of batch size.

Now let's analyse the range of numbers during the propagation of the network, it shouldn't be larger than 2^63(range of int64).
And for float64, it can only feel 10^-54 error so the range of float 64 in better bellow 10^54.
    1. float input can be scaled by a constant alpha(let's say 255(~=2^8), this factor only affect the precision of qinput data) and rounded to int then encrypt them
    2. an input tensor's range let's say be 0~2^8
    3. Encryption: get Sstar and Cstar, Sstar's range is 2^l*tBound so l should be less than 63/log2(tBound),about 53(if use float64, the added error term can't be precisely processed, it will behave like this: 2->0,107->128,388->384, the error caused becomes larger) M is almost with the same range of Sstar, and cPrime is M.dot(Cstar) almost at the same range with c(0~2^8*w = 0~2^45).
    4. Linear Transform: get Sprime = G.dot(S), then we perform keyswitch again, Range(Sstar(Sprime)) should be similar with Range(Sstar(S))(This Sprime should be dealt as float tensor because it's not too large, so rounding will cause great error. it's ok to get bit matrix with float tensor, so we will get a float tensor M, use it to get a new ciphertext), maybe 10 times larger, so range(M) is 2^(l+3)*tBound, to let it be smaller than 2^63, l should be smaller than 50, let's say 49.

    5. add: the 2 ciphertext add together, so range(c) is 2*2^45 , l should be larger than 46, let's take it 48. This means the batch normalization is quite important to make the feature map stay in a proper range.
From above analysis, take default initial condition defined in 
"""

class HE():
    
    def __init__(self,w = 2**19,
            aBound=1000,
            tBound=1000,
            eBound=1000,
            l=45,
            GPU=False
            ):
        if GPU == True:
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        self.w = w
        self.aBound = aBound
        self.tBound = tBound
        self.eBound = eBound
        self.l = l

    def cuda(self):
        self.device = 'cuda'

    def cpu(self):
        self.device = 'cpu'

    def getBitVector(self,vector):
        shape = vector.shape
        ans = torch.zeros([shape[0],self.l,*shape[1:]],dtype=vector.dtype,
                device=vector.device)
        mask = (2*(vector>=0).type(vector.dtype)-1)
        temp = vector*mask
        temp = temp.view(1,*temp.shape)
        for i in range(self.l): 
            if temp.max() == 0:# if temp is not all zero
                break
            next_temp = temp//2
            ans[:,i] = temp - next_temp * 2
            temp = next_temp
        ans = ans*mask.view(shape[0],1,*shape[1:])
        ans = ans.view(-1,*vector.shape[1:])
        return ans

    def getBitMatrix(self,matrix):
        power = torch.arange(self.l,dtype=matrix.dtype,device=matrix.device)
        power = 2**power
        shape = matrix.shape
        power = eval('power.view({}-1)'.format(matrix.dim()*'1,')) 
        ans = power*matrix.view(*shape,1)
        ans = ans.view(*shape[:-1],-1)
        return ans

    def getRandomMatrix(self, size, bound, device):
        return torch.randint(low = 0, high = bound,
                size=size, dtype=torch.float64,
                device=device)

    def keySwitchMatrix(self,s, T):
        Sstar = self.getBitMatrix(s)
        A = self.getRandomMatrix((T.shape[1],Sstar.shape[1]),self.aBound,s.device)
        E = self.getRandomMatrix((Sstar.shape[0],Sstar.shape[1]),self.eBound,s.device) 
        #pdb.set_trace()
        return torch.cat([Sstar+E-T.mm(A),A],0)

    def keySwitch(self,M,c):
        cstar = self.getBitVector(c)
        if cstar.dim != 2:
            shape = cstar.shape
            cstar = cstar.view(shape[0],-1)
            return M.mm(cstar).view(-1,*shape[1:])
        else:
            return M.mm(cstar)

    def getSecretkey(self,T):
        I = torch.eye(T.shape[0],dtype=T.dtype,device=T.device)
        return torch.cat([I,T],1)

    def encrypt(self,T,x):
        I = torch.eye(T.shape[0],dtype=T.dtype,device=T.device)
        return self.keySwitch(self.keySwitchMatrix(I,T), self.w*x)

    def decrypt(self,S,c,round=False):
        if c.dim != 2:
            shape = c.shape
            c = c.view(shape[0],-1)
            ans = S.mm(c).view(-1,*shape[1:])/self.w
            if round:
                return torch.round()
            return ans
        else:
            Sc = S.mm(c)/self.w
            if round:
                return torch.round(Sc)
            return Sc

    def linearTransformClient(self,G, S, T):
        return self.keySwitchMatrix(G.dot(S), T)

    def innerProd(self,c1,c2,M,verbose=False):
        if verbose:
            pass
        else:
            shape = c1.shape
            #pdb.set_trace()
            cc1 = c1.reshape(1,shape[0],-1)
            cc2 = c2.reshape(shape[0],1,-1)
            cc = (cc1*cc2).reshape(shape[0]**2,-1)
            cc = cc/self.w
            if c1.dim()!=2:
                return M.mm(self.getBitVector(cc)).view(-1,*c1.shape[1:])
            return M.mm(self.getBitVector(cc))

    def TGen(self,m,device):
        return self.getRandomMatrix([m,1],self.tBound,device)

    def innerProdClient(self,W,T):
        m = len(W)
        S = self.getSecretkey(T)
        W = W.transpose(0,1).contiguous().view(m,m*m)
        ans = S.t().mm(W).view(-1,m,m)
        ans = ans.transpose(0,1).contiguous().view(-1,m)
        ans = ans.mm(S).view(m,-1)
        return self.keySwitchMatrix(ans,T)
