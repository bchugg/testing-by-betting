from .base import AbstractBettor
from scipy.spatial.distance import pdist
from .utils import RBF_kernel
from .utils import deLaPena_martingale
from functools import partial
import numpy as np 

class TwoSampleScalarONS(AbstractBettor): 
    """Single dimensional two sample test using ONS betting strategy. 
    """

    def __init__(self, alpha=0.05) -> None:
        super().__init__(alpha, strategy='ONS')

    def predict(self, X1, X2):
        return X1 - X2
    

class KernelMMD(AbstractBettor): 
    """Kernel MMD betting strategy using ONS updates.

    Parameters (beyond AbstractBettor parameters):
    ----------
    kernel : callable, optional
        Kernel function, mapping two sets (arrays) of observations to matrix. 
        If None, use RBF kernel.
    post_processing : str, optional
        Transformation to apply to final value. Options are sinh, tanh, arctan, 
        and deLaPena martingale. If None, no transformation is applied.
    update_kernel_fn : callable, optional
        Function to update the user provided kernel. 
        If None, use default update function.

    Attributes (beyond AbstractBettor attributes)
    ----------
    kernel_name : str
        Either "rbf" (if default), or "other" if user has specified a kernel 
    X_hist, Y_hist : list of arrays
        Histories of observations 
    mmd_unnormalized : list of floats
        Histories of unnormalized MMD values (payoffs) 

    """

    def __init__(self, alpha=0.05, kernel=None, post_processing=None, 
                 update_kernel_fn=None, **kwargs) -> None:
        super().__init__(alpha, strategy='ONS', **kwargs)
        self.kernel = kernel 
        self.post_processing = post_processing
        self.bandwidth = 1
        
        # Default kernel is RBF. If another kernel is specified, 
        # check if update kernel function is provided and callable.  
        if kernel is None: 
            self.kernel = None
            self.kernel_name = 'rbf'
            self._update_kernel = self._update_rbf
        else: 
            self.kernel = kernel 
            self.kernel_name = 'other'
            if callable(update_kernel_fn): 
                self._update_kernel = update_kernel_fn
            else: 
                self._update_kernel = None
        
        # Histories
        self.X_hist = []
        self.Y_hist = []
        self.mmd_unnormalized = []

        post_processing_options = ['sinh', 'tanh', 'arctan', 'deLaPena', None]
        if post_processing not in post_processing_options: 
            raise ValueError(f'Post processing option {post_processing} not recognized. '+
                             'Options are {post_processing_options}')

    def _update_rbf(self):

        # Update bandwidth 
        bw = 1
        if len(self.X_hist) >= 20: 
            Z = np.concatenate((self.X_hist, self.Y_hist), axis=0)
            dists_ = pdist(Z)
            bw = np.median(dists_)

        # update the kernel
        self.kernel = partial(RBF_kernel, bw=bw)
    
    def predict(self, X1, X2):

        # Update history    
        self.X_hist.append(X1)
        self.Y_hist.append(X2)

        if len(self.X_hist) == 1: 
            return 0

        # Update kernel if applicable   
        self._update_kernel()
        assert self.kernel is not None, 'Kernel not yet set.'

        # Compute MMD
        KXX = self.kernel(self.X_hist, self.X_hist)
        KYY = self.kernel(self.Y_hist, self.Y_hist)
        KXY = self.kernel(self.X_hist, self.Y_hist)

        n0 = len(self.X_hist) - 1
        termX = np.mean((KXX[n0, :n0] - KXY[n0, :n0]))
        termY = np.mean((KXY[:n0, n0] - KYY[:n0, n0]))
        self.mmd_unnormalized.append(termX - termY)
        mmd = self.mmd_unnormalized[-1]
        
        # a heuristic that significantly improves the practical performance
        if len(self.X_hist) > 10:
            max_val = np.max(self.mmd_unnormalized)
            mmd = self.mmd_unnormalized[-1] / max_val 

        # Post processing
        if self.post_processing == 'sinh':
            mmd = np.sinh(mmd)
        elif self.post_processing == 'tanh':
            mmd = np.tanh(mmd)
        elif self.post_processing == 'arctan':
            mmd = (2/np.pi)*np.arctan(mmd)
        elif self.post_processing == 'deLaPena':
            mmd = deLaPena_martingale(mmd)
        
        return mmd
    