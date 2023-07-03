from .base import AbstractBettor
import numpy as np 

def sequential_experiment(bettor, X1, X2=None, verbose=True, **kwargs): 
    """Run sequential experiment with given bettor. Bettor can be either 
    a one-sample or two-sample test. 

    Parameters
    ----------
    bettor : AbstractBettor
        Bettor to use for sequential experiment. Bettor is initialized with 
        bettor(**kwargs)
    X1 : array-like
        First set of observations.
    X2 : array-like, optional
        Second set of observations. If None, run one-sample test.
    verbose : bool, optional    
        If True, print name of bettor before running experiment.
    **kwargs : dict, optional
        Additional keyword arguments to pass to bettor.

    Returns
    ---------
    wealth : list of floats 
        History of wealth. Length is either length of provided data (minimum of len(X1))
        and len(X2) or the time at which the null hypothesis is rejected. 
    reject_time : int or None
        Time at which null hypothesis is rejected. If None, null hypothesis was never 
        rejected.
    """
    
    assert issubclass(bettor, AbstractBettor), \
        "bettor must be an instance of AbstractBettor"
    
    test = bettor(**kwargs)
    if verbose: 
        print(f"Testing with {test.name}")
    
    n1 = len(X1)
    n2 = np.inf if X2 is None else len(X2)

    reject_time = None
    wealth = [] 
    for i in range(min(n1, n2)): 
        
        obs1 = X1[i]
        if X2 is not None:
            obs2 = X2[i]
            test.step(obs1, obs2)
        else: 
            test.step(obs1)

        wealth.append(test.wealth)
        if test.reject_null:
            reject_time = i
            break 

    return wealth, reject_time 
            
    
    