from .bets import OnlineNewtonStep
import numpy as np 


class AbstractBettor: 
    """Abstract class from which all strategies inherit. 
    Write your own strategy by inheriting from this class. 

    Properties:
        strategy: str
            betting strategy to use (sequence of lambdas)
        alpha: float
            significance level
        verbose: bool
            whether to print out rejection
        wealth: float
            current wealth
        iters: int
            number of iterations so far 
        randomize_ville: bool
            whether to use randomized ville inequality when 
            calculating if null hypothesis should be rejected (see Methods)
        reject_null: bool
            whether null hypothesis has been rejected
        wealth_history: list[float]
            history of wealth process
        bet_history: list[float]
            history of bets
        payoff_history: list[float]
            history of payoffs

    Methods:
        predict: function
            payoff function. Maps next observations(s) to payoff. 
            This needs to be implemented by each inheriting class.
        step: function
            Complete one iteration: calculate bet, payoff, and update wealth. 
        reject: function
            Compute if null hypothesis should be rejected by checking if 
            wealth exceeds 1/self.alpha. If self.randomized_ville is True, then, if process 
            has stopped, check if wealth exceeds u/self.alpha where u is a uniform random 
            variable. See Manole and Ramdas (2023), "Randomized and Exchangeable Improvements 
            of Markov's, Chebyshev's and Chernoff's Inequalities" for more details. 
    
    """

    def __init__(self, alpha=0.05, strategy='ONS', randomized_ville=True, verbose=True) -> None:
        self.strategy = strategy
        self.alpha = alpha
        self.verbose = verbose
        self.randomized_ville = randomized_ville

        # Initialize 
        self.wealth = 1
        self.iters = 0
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

        if stopped and self.randomized_ville: 
            u = np.random.uniform()
            cond = self.wealth >= u/self.alpha
        else: 
            cond = self.wealth >= 1/self.alpha
        
        self.reject_null = cond 
        return cond  
        
