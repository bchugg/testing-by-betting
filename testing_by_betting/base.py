from .bets import OnlineNewtonStep
import numpy as np 


class AbstractBettor: 
    """Abstract class from which all strategies inherit. 


    
    """

    def __init__(self, alpha=0.05, strategy='ONS', verbose=True) -> None:
        self.strategy = strategy
        self.alpha = alpha
        self.verbose = verbose

        # Initialize 
        self.wealth = 1
        self.iters = 1
        self.reject_null = False

        # Track histories
        self.wealth_history = []
        self.bet_history = []
        self.payoff_history = []

        if self.strategy == 'ONS': 
            self.betting_strategy = OnlineNewtonStep()
        else: 
            raise ValueError('Betting strategy unrecognized') 


    def predict(self): 
        raise NotImplementedError
     
    def step(self, *args): 
        
        # Compute next bet
        self.bet = self.betting_strategy.next_bet(self.payoff_history)
        self.bet_history.append(self.bet)
    
        # Calculate payoff 
        payoff = self.predict(*args)
        self.payoff_history.append(payoff)
        
        # Update wealth 
        self.wealth *= 1 + self.bet*payoff
        self.wealth_history.append(self.wealth)
        self.iters += 1

        # Check for rejection
        if self.reject(stopped=False):
            if self.verbose: 
                print(f'Reject null. Iters: {self.iters}')
    
    def reject(self, stopped=False): 

        if stopped: 
            u = np.random.uniform()
            cond = self.wealth >= u/self.alpha
        else: 
            cond = self.wealth >= 1/self.alpha
        
        self.reject_null = cond 
        return cond  
        
