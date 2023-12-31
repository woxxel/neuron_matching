import math
import numpy as np
import scipy as sp

phi = lambda x : 1/np.sqrt(2*np.pi)*np.exp(-1/2*x**2)
psi = lambda x : 1/2*(1 + sp.special.erf(x/np.sqrt(2)))

functions = {
      
      ## define some possible fitting functions
    'lognorm': lambda x,sigma,mu : 1/(x*sigma*np.sqrt(2*np.pi))*np.exp(-(np.log(x)-mu)**2/(2*sigma**2)),
    
    'lognorm_reverse': lambda x,sigma,mu : 1/((1-x)*sigma*np.sqrt(2*np.pi))*np.exp(-(np.log(1-x)-mu)**2/(2*sigma**2)),
   
    'lognorm_shifted': lambda x,sigma,mu,s : 1/((x+s)*sigma*np.sqrt(2*np.pi))*np.exp(-(np.log(x+s)-mu)**2/(2*sigma**2)),
    
    'lognorm_reverse_shifted': lambda x,sigma,mu,s : 1/((-x+s)*sigma*np.sqrt(2*np.pi))*np.exp(-(np.log(-x+s)-mu)**2/(2*sigma**2)),
    
    'gauss': lambda x,sigma,mu : 1/np.sqrt(2*np.pi*sigma**2)*np.exp(-(x-mu)**2/(2*sigma**2)),


    
    'truncated_lognorm': lambda x,sigma,mu : 1/sigma * phi((x-mu)/sigma) / (psi((1-mu)/sigma) - psi((0-mu)/sigma)),
    
    'truncated_lognorm_reverse': lambda x,sigma,mu : 1/sigma * phi((1-x-mu)/sigma) / (psi((1-mu)/sigma) - psi((0-mu)/sigma)),



    'beta': lambda x,a,b : x**(a-1)*(1-x)**(b-1) / (math.gamma(a)*math.gamma(b)/math.gamma(a+b)),

    'linear_sigmoid': lambda x,m,sig_slope,sig_center : m*x/(1+np.exp(-sig_slope*(x-sig_center))),
}
