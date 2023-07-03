from scipy.spatial.distance import cdist
import numpy as np 

def RBF_kernel(x, y=None, bw=1.0):
    """Compute squared exponential (or RBF) kernel matrix"""
    y = x if y is None else y 
    dists = cdist(x, y, 'euclidean') # Matrix of pairwise distances
    sq_dists = dists * dists 
    K = np.exp(-sq_dists/(2*bw*bw))
    return K

def linear_kernel(x, y=None):
    y = x if y is None else y 
    K = np.einsum('ji, ki ->jk', x, y) 
    return K 

def polynomial_kernel(x, y=None, c=1.0, p=2):
    L = linear_kernel(x, y) 
    K = (c + L)**p 
    return K 

def deLaPena_martingale(val): 
    val_plus = np.exp(val - val*val/2)
    val_minus = np.exp(-val - val*val/2)
    if val_plus >= val_minus:
        return val_plus - 1
    else:
        return 2 - val_minus - 1