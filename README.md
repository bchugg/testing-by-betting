![cover image](/assets/testing_by_betting.png)

# Sequential, nonparametric hypothesis testing 

This package implements various procedures for **[sequential, nonparametric hypothesis testing](https://en.wikipedia.org/wiki/Sequential_analysis)** by employing strategies from [game-theoretic probability and statistics](https://arxiv.org/pdf/2210.01948.pdf). This is known as testing by betting. 

The sequential nature of the tests enable practitioners to monitor data as they arrive and to continually test the hypothesis of interest. This is in contrast to fixed-time hypothesis tests which do not allow for continual testing, and which necessitate choosing the sample size beforehand. 

**Contents** 
- [Installation](#installation)
- [Usage](#usage)
- [Background](#background)
- [Implemented tests](#strategies)
- [References](#bib)


# 🛠️ Installation <a id='installation'></a>

Run 

```pip install testing-by-betting```

Or 

```python3 -m pip install testing-by-betting```

# 🤖 Usage <a id='usage'></a>

First, import the desired test from either `testing_by_betting.onesample` or `testing_by_betting.twosample`. Here we use the KernelMMD test of Shekhar and Ramdas (2023). 

```python
from testing_by_betting.twosample import KernelMMD
from scipy.stats import multivariate_normal
```

Here we present the basic functionality. More advanced illustrations of the methods can be found in `demos/`. 

### 🔩 Getting started

All tests interact with the environment via a `step` method, which takes one or more observations at a time (depending on the test). This simulates how sequential tests can be used in practice; we need not wait a batch of data to arrive. Here we generate iid observations from two multivariate Gaussians.  The following typically rejects after ~150-200 samples. 

```python
kernel_mmd = KernelMMD(alpha=0.05)

while not kernel_mmd.reject(): # Stop whenever test rejects; otherwise continue. 
   
    # Generate new data 
    X1 = multivariate_normal.rvs(mean=[0,0,0], size=1)
    X2 = multivariate_normal.rvs(mean=[0,1,0], size=1)

    # Perform the sequential test   
    kernel_mmd.step(X1, X2)    
```

Here, `alpha` is the signifiance level: the probability that the test _ever_ rejects under the null is `alpha`. 


### 🧪 Batch testing

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

### ⏲️ Parallel processing 

TODO. 

### 🏁 Baselines 

TODO. 

# 📚 Background <a id='background'></a>

The idea at the heart of testing by betting is to view sequential hypothesis testing as an iterated game between 
two players, who we call nature and skeptic.  The game proceeds as follows: 

- At time 0, the skeptic begins with a wealth of $\mathcal{K}_0=1$. 
- At time $t$, the skeptic chooses a payoff function $S_t:\mathcal{Z} \to [0,\infty]$. Crucially, the the payoff function must obey $\mathbb{E}\_P[S\_t(Z)|\mathcal{F}\_{t-1}] \leq 1$ for all $P$ in the null hypothesis $H_0$. (Composite nulls are thus allowed.) 
- Nature then reveals a value $Z_t$ (which might consist of a single scalar observation, two multivariate observations, etc). 
- Skeptic updates wealth as $\mathcal{K}\_t = \mathcal{K}\_{t-1}\cdot S\_t(Z_t)$. 

The skeptic's wealth is used as a measure of evidence against the null: if it grows too large we reject. In particular, for a given significance level $\alpha$, we reject the null as soon as $\mathcal{K}\_t \geq 1/\alpha$. Mathematically, this is guaranteed to be a level $\alpha$ sequential test by Ville's inequality for nonnegative supermartingales. See [this survey](https://arxiv.org/pdf/2210.01948.pdf) for more details. 


# 🧰 Implemented Strategies <a id='strategies'></a>

This package implements various betting strategies, $S_t$, for the skeptic (see [Background](#background)). We implement: 

### `OneSampleScalarONS` 
This uses the strategy $S_t(Z) = 1 + \lambda_t( Z_t - \mu_0)$ where $Z\in\mathbb{R}$, and tests the simple null $H_0: \mathbb{E}[Z]=\mu_0$ against $H_1: \mathbb{E}[Z]\neq \mu_0$. ONS refers to "Online Newton Step," which is the strategy we employ to choose the sequence $\{\lambda_t\}$. 

### `TwoSampleScalarONS`
Similar to above, but conducts a sequential _two-sample_ test. Here $Z_t=(X_t,Y_t)$ and we use $S_t(X,Y) = 1 + \lambda_t(X_t - Y_t)$. Again, $\lambda_t$ is chosen via ONS. This tests $H_0: \mathbb{E}[X] = \mathbb{E}[Y]$ vs $H_1: \mathbb{E}[X] \neq \mathbb{E}[Y]$. 


### `ScalarKS`
A scalar test based on comparing the empirical CDFs. 

### `KernelMMD` 
A two-sample test for multivariate data $Z_t=(X_t,Y_t)$. We use $S_t(X_t,Y_t) = 1 + \lambda_t(K(X_t) - K(Y_t))$ for some kernel $K$. The default kernel is the GaussianRBF, but the user may specify others. $\lambda_t$ is chosen via ONS.  


# 📖 Reading list <a id='bib'></a>

This package implements several methods discussed in the following papers: 

- [Nonparametric Two-Sample Testing by Betting](https://arxiv.org/abs/2112.09162), by Shekhar and Ramdas. _IEEE Transactions on Information Theory_, 2023. 
    - Proposes the `KernelMDD` test above. 
- [Sequential estimation of quantiles with applications to A/B-testing and best-arm identification](https://arxiv.org/abs/1906.09712) by Howard and Ramdas. _Bernoulli, 2022_.  
    - Proposes the `ScalarKS` test. 
- [Sequential kernelized independence testing](https://arxiv.org/pdf/2212.07383.pdf), by Podkopaev et al. _ICML_ 2023. 
- [Sequential predictive two-sample and independence testing](https://arxiv.org/pdf/2305.00143.pdf), by Podkopaev and Ramdas. Preprint. 
- [Comparing Sequential Forecasters](https://arxiv.org/pdf/2110.00115.pdf), by Choe and Ramdas. Preprint. 
- [Estimating means of bounded random variables by betting](https://arxiv.org/pdf/2010.09686.pdf), by Waudby-Smith and Ramdas. _JRSSB 2023, discussion paper_. 
    - Inverts one sample tests to achieve time-uniform confidence sequences for the mean of random variables.  
- [Game-Theoretic Statistics and
Safe Anytime-Valid Inference](https://arxiv.org/pdf/2210.01948.pdf), by Ramdas et al. _Statistical Science (2023)_. 
    - Survey of game-theoretic probability and sequential hypothesis testing. 
