# Testing by betting

This package implements various procedures for **[sequential, nonparametric hypothesis testing](https://en.wikipedia.org/wiki/Sequential_analysis)** by employing strategies from [game-theoretic probability](https://arxiv.org/pdf/2210.01948.pdf). 

The sequential nature of the tests enable practitioners to monitor data as they arrive and to continually test the hypothesis of interest. This is in contrast to fixed-time hypothesis tests which do not allow for continual testing, and which necessitate choosing the sample size beforehand. 



# Installation 

Run 

```pip install testing-by-betting```

Or 

```python3 -m pip install testing-by-betting```

# Usage 

First, import the desired test from either `testing_by_betting.onesample` or `testing_by_betting.twosample`. Here we use the KernelMMD test of Shekhar and Ramdas (2023). 

```python
from testing_by_betting.twosample import KernelMMD
from scipy.stats import multivariate_normal
```

All tests interact with the environment via a `step` method, which takes one or more observations at a time (depending on the test). This simulates how sequential tests can be used in practice; we need not wait a batch of data to arrive. Here we generate iid observations from two multivariate Gaussians. The following typically rejects after ~150-200 samples.  

```python
kernel_mmd = KernelMMD(alpha=0.05)

while not kernel_mmd.reject(): # Stop whenever test rejects; otherwise continue. 
   
    # Generate new data 
    X1 = multivariate_normal.rvs(mean=[0,0,0], size=1)
    X2 = multivariate_normal.rvs(mean=[0,1,0], size=1)

    # Perform the sequential test   
    kernel_mmd.step(X1, X2)    
```

To make research and experimentation easier, you can also provide all the data at once to the method 
`sequential_experiment`. 
This can take either two sequences of observations or only one (by omitting `X2`). 
Extra arguments (in this case `alpha` and `post_processing`) will be passed to the betting strategy when it is instantiated.  

```python
from testing_by_betting.experiment import sequential_experiment

X1 = multivariate_normal.rvs(mean=[0,0,0], size=300)
X2 = multivariate_normal.rvs(mean=[0,1,0], size=300)

wealth_history, rejection_time = sequential_experiment(KernelMMD, X1, X2, alpha=0.01, post_processing='arctan')
```

More advanced illustrations of the various methods can be found in `demos/`. 


# Implemented Strategies 


# Reading list 

This package implements several methods discussed in the following papers: 

- [Nonparametric Two-Sample Testing by Betting](https://arxiv.org/abs/2112.09162), by Shekhar and Ramdas. _IEEE Transactions on Information Theory_, 2023. 
- [Sequential kernelized independence testing](https://arxiv.org/pdf/2212.07383.pdf), by Podkopaev et al. _ICML_ 2023. 
- [Sequential predictive two-sample and independence testing](https://arxiv.org/pdf/2305.00143.pdf), by Podkopaev and Ramdas. Preprint. 
- [Comparing Sequential Forecasters](https://arxiv.org/pdf/2110.00115.pdf), by Choe and Ramdas. Preprint. 
- [Estimating means of bounded random variables by betting](https://arxiv.org/pdf/2010.09686.pdf), by Waudby-Smith and Ramdas. _JRSSB 2023, discussion paper_. 
- [Game-Theoretic Statistics and
Safe Anytime-Valid Inference](https://arxiv.org/pdf/2210.01948.pdf), by Ramdas et al. _Statistical Science (2023)_. 