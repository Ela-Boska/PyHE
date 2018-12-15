import numpy as np
import decimal,math
decimal.getcontext().prec = 33
double = np.vectorize(decimal.Decimal,[object])
floor = np.vectorize(math.floor,[object])
import pdb
import time

r"""
The followings are the error caused by operations:
    encrypt:            eBound
    scale:              1/6
    linear transform    eBound
    add                 eBound
    inner product       m^(1/2) * tBound /6
So for a network with 20 layers, asssume the shape of its feature map is 32*128*128, the error caused is
about 2^28*2^3, so to be save, let's make w be 2^6 times as large as 2^31, 2^37. 
Note that the error caused by the network is proportional to the depth of layers and the sqaure root of 
the size of its feature map.

Now let's analyse the range of numbers during the propagation of the network, it shouldn't be larger than 2^63(range of int64).
And for float64, it can only feel 10^-54 error so the range of float 64 in better bellow 10^54.
    1. float input can be scaled by a constant alpha(let's say 255(~=2^8), this factor only affect the precision of input data) and 
       rounded to int then encrypt them
    2. a input tensor's range let's say be 0~2^8
    3. Encryption: get Sstar and Cstar, Sstar's range is 2^l*tBound so l should be less than 63/log2(tBound),about 53. M is almost with the 
       same range of Sstar, and cPrime is M.dot(Cstar) almost at the same range with c(0~2^8*w = 0~2^45).
    4. Linear Transform: get Sprime = G.dot(S), then we perform keyswitch again, Range(Sstar(Sprime)) should be similar with 
       Range(Sstar(S)), maybe 10 times larger, so range(M) is 2^(l+3)*tBound, to let it be smaller than 2^63, l should be smaller than 50, 
       let's say 49
    5. add: the 2 ciphertext add together, so range(c) is 2*2^45 , l should be larger than 46, let's take it 48. This means the 
       batch normalization is quite important to make the feature map stay in a proper range.
From above analysis, take default initial condition defined in 


"""

def round(tensor):
    Floor = floor(tensor)
    delta = tensor - Floor
    return Floor + (delta>=0.5)

class HE():

    def __init__(self,w = 2**40,
            aBound=1000,
            tBound=1000,
            eBound=1000,
            l=100):
        self.w = w
        self.aBound = aBound
        self.tBound = tBound
        self.eBound = eBound
        self.l = l

    def getBitVector(self,vector):
        shape = vector.shape
        dim = vector.ndim
        ans = np.zeros([*shape,self.l],dtype=object)
        mask = 2*(vector>=0)-1
        temp = vector*mask
        for i in np.arange(self.l): 
            if temp.max() == 0:# if temp is not all zero
                break
            next_temp = temp//2
            exec('ans[{}i] = temp - next_temp * 2'.format(dim*':,'))
            temp = next_temp
        ans = ans*mask.reshape(*vector.shape,1)
        ans = ans.reshape(*vector.shape[:-1],-1)
        return ans

    def getBitMatrix(self,S):
        rows, cols = S.shape
        ans = np.zeros((rows,cols,self.l),dtype=object)
        # ans dtype is python int object without the range limit
        times = (2**np.arange(self.l).astype(object)).reshape(1,1,-1)
        ans = S.reshape(*S.shape,1)*times
        return ans.reshape(S.shape[0],-1)

    def keySwitch(self,M,c,batching=False,verbose=False):
        if not batching:
            cstar = self.getBitVector(c)
            return M.dot(cstar)
        else:
            if verbose:
                t1 = time.time()
                cstar = self.getBitVector(c)
                t2 = time.time()
                ans = cstar.dot(M.T)
                t3 = time.time()
                #pdb.set_trace()
                info = r'''time spent on creating bit vector: {}s
time spent on doting: {}s'''.format(t2-t1,t3-t2)
                print(info)
                return ans

            cstar = self.getBitVector(c)
            return cstar.dot(M.T)

    def getRandomMatrix(self,row, col, bound):
        return np.random.randint(low = 0, high = bound, size = (row,col))

    def getSecretkey(self,T):
        I = np.eye(T.shape[0]).astype(int).astype(object)
        return np.concatenate([I,T],1)

    def decrypt(self,S ,c,batching=False):
        if not batching:
            Sc = S.dot(c)
        else:
            Sc = c.dot(S.T)
        return round(double(Sc)/self.w)

    def keySwitchMatrix(self,s, T):
        Sstar = self.getBitMatrix(s)
        A = self.getRandomMatrix(T.shape[1],Sstar.shape[1],self.aBound)
        E = self.getRandomMatrix(Sstar.shape[0],Sstar.shape[1],self.eBound) 
        return np.concatenate([Sstar+E-T.dot(A),A],0)

    def encrypt(self,T,x,batching=False,verbose=False):
        if verbose:
            t1 = time.time()
            I = np.eye(x.shape[-1]).astype(int).astype(object)
            t2 = time.time()
            M = self.keySwitchMatrix(I,T)
            t3 = time.time()
            ans = self.keySwitch(M, self.w*x,batching=batching,verbose=True)
            t4 = time.time()
            #pdb.set_trace()
            info = r"""time spent on creating I: {}s
time spent on calculating M: {}s
time spent on key switching: {}s""".format(t2-t1,t3-t2,t4-t3)
            print(info)
            return ans

        I = np.eye(x.shape[-1]).astype(int).astype(object)
        return self.keySwitch(self.keySwitchMatrix(I,T), self.w*x,batching=batching)

    def linearTransform(self,M,c,batching=False):
        if not batching:
            return M.dot(self.getBitVector(c))
        else:
            return self.getBitVector(c).dot(M.T)

    def linearTransformClient(self,G, S, T):
        return self.keySwitchMatrix(G.dot(S), T)

    def innerProd(self,c1,c2,M,verbose=False):
        if verbose:
            t0 = time.time()
            shape = c1.shape
            cc1 = c1.reshape(*shape,1)
            cc2 = c2.reshape(*shape[:-1],1,-1)
            cc = (cc1*cc2).reshape(*shape[:-1],-1)
            t1 = time.time()
            #pdb.set_trace()
            cc = round((cc/self.w))
            t2 = time.time()
            #pdb.set_trace()
            ans = (self.getBitVector(cc)).dot(M.T)
            t3 = time.time()
            info = r'''time spent on reshaping: {}s
time spent on rounding: {}s
time spent on doting: {}s'''.format(t1-t0,t2-t1,t3-t2)
            print(info)
            return ans

        shape = c1.shape
        #pdb.set_trace()
        cc1 = c1.reshape(*shape,1)
        cc2 = c2.reshape(*shape[:-1],1,-1)
        cc = (cc1*cc2).reshape(*shape[:-1],-1)
        cc = round((cc/self.w))
        return (self.getBitVector(cc)).dot(M.T)

    def keyGen(self,m,n=None):
        if n == None:
            n = m
        S = self.getRandomMatrix(m,n,100)
        return S

    def TGen(self,m,delta=1):
        return self.getRandomMatrix(m,delta,self.tBound)

    def WeightedInnerProdClient(self,W,T):
        # W: a matrix of shape (m,m,m) 
        # ans[i] = xT W[i] x
        S = self.getSecretkey(T)
        #pdb.set_trace()
        ans = S.T.dot(W.transpose(1,0,2))
        ans = ans.transpose(1,0,2)
        ans = ans.dot(S)
        ans = ans.reshape(ans.shape[0],-1)
        return self.keySwitchMatrix(ans,T)