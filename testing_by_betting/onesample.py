from .base import AbstractBettor

class SingleDimensionONS(AbstractBettor): 

    def __init__(self, mu, alpha=0.05) -> None:
        super().__init__(alpha, strategy='ONS')
        self.mu = mu 

    def predict(self, X1):
        return X1 - self.mu