import math
import numpy as np
from scipy.special import erf
from scipy.stats import skewnorm as skewed_gauss

phi = lambda x : 1/np.sqrt(2*np.pi)*np.exp(-1/2*x**2)
psi = lambda x : 1/2*(1 + erf(x/np.sqrt(2)))

functions = {
      
      ## define some possible fitting functions
    'lognorm': lambda x,m,sigma,mu : m * 1/(x*sigma*np.sqrt(2*np.pi))*np.exp(-(np.log(x)-mu)**2/(2*sigma**2)),
    
    'lognorm_reverse': lambda x,m,sigma,mu : m * 1/((1-x)*sigma*np.sqrt(2*np.pi))*np.exp(-(np.log(1-x)-mu)**2/(2*sigma**2)),
   
    'lognorm_shifted': lambda x,m,sigma,mu,s : m * 1/((x+s)*sigma*np.sqrt(2*np.pi))*np.exp(-(np.log(x+s)-mu)**2/(2*sigma**2)),
    
    'lognorm_reverse_shifted': lambda x,m,sigma,mu,s : m * 1/((-x+s)*sigma*np.sqrt(2*np.pi))*np.exp(-(np.log(-x+s)-mu)**2/(2*sigma**2)),
    
    'gauss': lambda x,m,sigma,mu : m * 1/np.sqrt(2*np.pi*sigma**2)*np.exp(-(x-mu)**2/(2*sigma**2)),

    'skewed_gauss': lambda x,m,sigma,mu,skew=-1. : m * skewed_gauss.pdf(x,skew,sigma,mu),

    'gamma': lambda x,m,a,b : m * x**(a-1)*np.exp(-x/b)*b**a/(math.gamma(a)),
    
    'gamma_reverse': lambda x,m,a,b : m * (1-x)**(a-1)*np.exp(-(1-x)/b)*b**a/(math.gamma(a)),

    
    'truncated_lognorm': lambda x,m,sigma,mu : m * 1/sigma * phi((x-mu)/sigma) / (psi((1-mu)/sigma) - psi((0-mu)/sigma)),
    
    'truncated_lognorm_reverse': lambda x,m,sigma,mu : m * 1/sigma * phi((1-x-mu)/sigma) / (psi((1-mu)/sigma) - psi((0-mu)/sigma)),



    'beta': lambda x,m,a,b : m * x**(a-1)*(1-x)**(b-1) / (math.gamma(a)*math.gamma(b)/math.gamma(a+b)),

    'linear_sigmoid': lambda x,m,sig_slope,sig_center : m * x/(1+np.exp(-sig_slope*(x-sig_center))),
    # 'linear_sigmoid': lambda x,sig_slope,sig_center : x/(1+np.exp(-sig_slope*(x-sig_center))),
}
