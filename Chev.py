import numpy as np
from scipy import integrate
import torch
import math
import matplotlib.pyplot as plt

class polynomial():
    
    def __init__(self, coe):
        self.coe = np.array(coe)
        self.degree = len(coe)-1

    def __len__(self):
        return len(self.coe)

    def __call__(self,x):
        if isinstance(x,np.ndarray):
            dim = x.ndim
            x = x.reshape(*x.shape,1)
            shape = [1]*dim + [-1]
            return np.sum(x**np.arange(self.degree+1).reshape(shape) * self.coe,-1)
        elif isinstance(x,torch.Tensor):
            coe = torch.from_numpy(self.coe).type(x.dtype).device(x.device)
            dim = x.dim()
            x = x.view(*x.shape,1)
            shape = [1]*dim + [-1]
            return torch.sum(x**np.arange(self.degree+1,dtype=x.dtype,device=x.device).reshape(shape) * coe.view(shape),-1)
        else:
            return np.sum(x**np.arange(self.degree+1)*self.coe)
    
    def __repr__(self):
        return "ploynomial with coefficients, {}".format(self.coe.tolist())

    def integrate(self,start=0):
        ans = self.coe
        ans = ans/np.arange(1,len(ans)+1)
        ans = [0] + ans.tolist()
        ans = polynomial(ans)
        ans.coe[0] = -ans(start)
        return ans


def _approximate(fc,fc_strech,stdnorm,start,end):
    denominator = integrate.quad(lambda x:stdnorm(x)*fc_strech(x)**2,start,end)[0]
    numerator = integrate.quad(lambda x:stdnorm(x)*fc_strech(x)*fc(x),start,end)[0]
    return numerator/denominator


class chev():

    T0 = polynomial([1])
    T1 = polynomial([0,1])
    stdnorm_standard = lambda x: (1-x**2)**(-0.5)
    stdnorm_modified = lambda x: math.exp(-1/(1e-5+x**2))

    def __init__(self, degree=1):
        self.degree = degree
        if degree == 0:
            self.polynomials = [chev.T0]
        else:
            self.polynomials = [chev.T0,chev.T1]
            for i in range(0,degree-1):
                coe1 = self.polynomials[-1].coe.tolist()
                coe1 = [0]+coe1
                coe1 = np.array(coe1)*2
                coe2 = self.polynomials[-2].coe.tolist()
                coe2 = coe2 + 2*[0]
                coe2 = np.array(coe2)
                coe = coe1 - coe2
                self.polynomials.append(polynomial(coe))
        
    def __repr__(self):
        return 'a set of the first {} chev polynomials'.format(self.degree+1)

    def approximate(self,function,standard, output_form):
        ans = []
        if standard:
            for i in range(len(self.polynomials)):
                ans.append(_approximate(function,self.polynomials[i],chev.stdnorm_standard,-1,1))
        else:
            for i in range(len(self.polynomials)):
                ans.append(_approximate(function,self.polynomials[i],chev.stdnorm_modified,-1,1))
        if output_form == 'chev':
            return np.array(ans)
        elif output_form == 'x':
            def _compensate(poly):
                length = len(self.polynomials[-1])
                coe = poly.coe.tolist()
                coe = coe + (length-len(coe))*[0]
                return coe
            coes = list(map(_compensate,self.polynomials)) 
            coes = np.array(coes)
            coes = np.array(ans).reshape(-1,1)*coes
            return coes.sum(0)
def relu(x):
    if isinstance(x,np.ndarray):
        return (x+np.absolute(x))/2
    return max(0,x)

def sigmoid(x):
    if isinstance(x,np.ndarray):
        exp = x.exp()
        return exp/(1+exp)
    else:
        exp = math.exp(x)
        return exp/(1+exp)

def main(degree):
    a = chev(degree)
    x = np.linspace(-1,1,1000)
    y1 = relu(x)
    poly1 = polynomial(a.approximate(relu,True,'x'))
    poly2 = polynomial(a.approximate(relu,False,'x'))
    poly3 = polynomial(a.approximate(sigmoid,True,'x')).integrate(-1)
    poly4 = polynomial(a.approximate(sigmoid,False,'x')).integrate(-1)
    #y2 = poly1(x)
    #y3 = poly2(x)
    y4 = poly3(x)
    y5 = poly4(x)
    y6 = (x+1)**2/4
    plt.plot(x,y1,label='relu')
    #plt.plot(x,y2,label='standard-direct-relu')
    #plt.plot(x,y3,label='modified-direct-relu')
    plt.plot(x,y4,label='standard')
    plt.plot(x,y5,label='modified')
    plt.plot(x,y6,label='(1+x)^2/4')
    plt.title('approximation of relu of degree {}'.format(degree))
    plt.legend()
    plt.show()


if __name__=='__main__':
    import sys
    degree = int(sys.argv[1])
    main(degree)