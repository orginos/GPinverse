""" PyTorchPoly """
# Taken from https://github.com/goroda/PyTorchPoly/blob/master/poly.py

import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.special import gammaln

# import pyindex

def legendre(x, degree):
    retvar = torch.ones(x.size(0), degree+1).type(x.type())
    # retvar[:, 0] = x * 0 + 1
    if degree > 0:
        retvar[:, 1] = x
        for ii in range(1, degree):
            retvar[:, ii+1] = ((2 * ii + 1) * x * retvar[:, ii] - \
                               ii * retvar[:, ii-1]) / (ii + 1)
    return retvar

def legendre_01(y, degree):
    x = 2*y-1
    retvar = torch.ones(x.size(0), degree+1).type(x.type())
    # retvar[:, 0] = x * 0 + 1
    if degree > 0:
        retvar[:, 1] = x
        for ii in range(1, degree):
            retvar[:, ii+1] = ((2 * ii + 1) * x * retvar[:, ii] - \
                               ii * retvar[:, ii-1]) / (ii + 1)
    return retvar

def chebyshev(x, degree):
    retvar = torch.zeros(x.size(0), degree+1).type(x.type())
    retvar[:, 0] = x * 0 + 1
    if degree > 0:
        retvar[:, 1] = x
        for ii in range(1, degree):
            retvar[:, ii+1] = 2 * x * retvar[:, ii] -  retvar[:, ii-1]

    return retvar

def hermite(x, degree):
    retvar = torch.zeros(x.size(0), degree+1).type(x.type())
    retvar[:, 0] = x * 0 + 1
    if degree > 0:
        retvar[:, 1] = x
        for ii in range(1, degree):
            retvar[:, ii+1] = x * retvar[:, ii] - retvar[:, ii-1] / ii

    return retvar

# I am not sure if Jacobi's are implemented correctly
# I need to check against my old Tensorflow version
def jacobi(x, degree,alpha,beta):
    retvar = torch.ones(x.size(0), degree+1).type(x.type())
    if degree > 0:
        retvar[:, 1] = alpha+1 + 0.5*(alpha+beta+2)*(x-1)
        for ii in range(1, degree):
            n = ii+1
            a = n+alpha
            b = n+beta
            c = a + b 
            retvar[:, n] = (c -1)*( c*(c-2)*x + (a-b)*(c-2*n))*retvar[:, ii] - 2.0*(a-1)*(b-1)*c*retvar[:, ii-1]
            retvar[:, n] /= 2*n*(c-n)*(c-2) 

    return retvar
def jacobi_norm(degree,alpha,beta):
    n = torch.arange(0,degree+1)
    a = n+alpha
    b = n+beta
    c = a + b 
    norm = 2**(alpha+beta+1)/(c+1)*torch.exp(gammaln(a+1)+gammaln(b+1) - gammaln(c+n+1) -gammaln(n+1))
    return norm

def jacobi_01(x,degree,alpha,beta):
    y = 1- 2*x 
    return jacobi(y,degree,alpha,beta)

def jacobi_weight(x,alpha,beta):
    return (1 - x)**alpha * (1+x)**beta

def jacobi_01_weight(x,alpha,beta):
    return jacobi_weight(1- 2*x,alpha,beta)


def jacobi_01_norm(degree,alpha,beta):
#    foo = jacobi_norm(degree,alpha,beta)
#    print(foo)
    return jacobi_norm(degree,alpha,beta)*2#torch.tensor([2**(alpha+beta+1)])

class UnivariatePoly(nn.Module):
    """ Univariate Legendre Polynomial """
    def __init__(self, PolyDegree, poly_type):
        super(UnivariatePoly, self).__init__()
        self.degree = PolyDegree
        self.linear = nn.Linear(PolyDegree+1, 1, bias=False)
        self.poly_type = poly_type

    def forward(self, x):

        if self.poly_type == "legendre":
            vand = legendre(x, self.degree)
        elif self.poly_type == "chebyshev":
            vand = chebyshev(x, self.degree)
        elif self.poly_type == "hermite":
            vand = hermite(x, self.degree)            
        else:
            print("No Polynomial type ", self.poly_type, " is implemented")
            exit(1)
        # print("vand = ", vand)
        retvar = self.linear(vand)

        return retvar
