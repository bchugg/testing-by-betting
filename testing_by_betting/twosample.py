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
    """Kernel MMD betting strategy."""

    def __init__(self, alpha=0.05, kernel=None, post_processing=None, **kwargs) -> None:
        super().__init__(alpha, strategy='ONS', **kwargs)
        self.kernel = kernel 
        self.post_processing = post_processing
        self.bandwidth = 1

        self.X_hist = []
        self.Y_hist = []

        post_processing_options = ['sinh', 'tanh', 'arctan', 'deLaPena', None]
        if post_processing not in post_processing_options: 
            raise ValueError(f'Post processing option {post_processing} not recognized. '+
                             'Options are {post_processing_options}')



    def _update_kernel(self):

        # Update bandwidth 
        Z = np.concatenate((self.X_hist, self.Y_hist), axis=0)
        dists_ = pdist(Z)
        # obtain the median of these pairwise distances 
        bw = np.median(dists_)
        # update the kernel
        self.kernel = partial(RBF_kernel, bw=bw)
    

    def predict(self, X1, X2):

        # Update history    
        self.X_hist.append(X1)
        self.Y_hist.append(X2)

        # Update kernel 
        self._update_kernel

        # Compute MMD
        KXX = self.kernel(self.X_hist, self.X_hist)
        KYY = self.kernel(self.Y_hist, self.Y_hist)
        KXY = self.kernel(self.X_hist, self.Y_hist)

        termX = np.mean((KXX[-1, :] - KXY[-1, :]))
        termY = np.mean((KXY[:, -1] - KYY[:, -1]))
        self.mmd_unnormalized = termX - termY
        mmd = self.mmd_unnormalized
        
        ### a heuristic that significantly improve the practical performance
        if len(self.X_hist) > 10:
            max_val = np.max(self.mmd_unnormalized)
            mmd = self.mmd_unnormalized / max_val 

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
