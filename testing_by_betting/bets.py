import numpy as np 

class AbstractBet: 
    """Abstract class from which all betting strategies inherit.
    Write your own betting strategy by inheriting from this class.

    Methods: 
        next_bet: function - return next bet (a scalar value) given payoff history
    """

    def __init__(self) -> None:
        pass

    def next_bet(self, payoff_history): 
        raise NotImplementedError


class OnlineNewtonStep(AbstractBet):
    """Online Newton Step betting strategy. 
    See Shekhar and Ramdas (2023), "Nonparametric Two-Sample Testing by Betting"
    https://arxiv.org/pdf/2112.09162.pdf for more details. 
    """

    def __init__(self) -> None:
        self.const = 2 / (2 - np.log(3))
        self.sum_z_squared = 1 
        self.prev_lambd = 0

    def next_bet(self, payoff_history):

        if len(payoff_history) == 0: 
            return 0 # bet 0 for the first time 
        else: 
            prev_payoff = payoff_history[-1]
            z = -prev_payoff / (1 + self.prev_lambd*prev_payoff)
            self.sum_z_squared += z**2
            lambd = max(
                min(
                    self.prev_lambd - self.const*z/self.sum_z_squared, 1/2
                    ), 
                -1/2
                )
            self.prev_lambd = lambd 
            return lambd 
