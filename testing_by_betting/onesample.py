from .base import AbstractBettor

class OneSampleScalarONS(AbstractBettor): 
    """Single dimensional one sample test using ONS betting strategy.
    """

    def __init__(self, mu, alpha=0.05) -> None:
        super().__init__(alpha, strategy='ONS')
        self.mu = mu 

    @property
    def name(self):
        return 'Scalar one sample test with ONS'

    def predict(self, X1):
        return X1 - self.mu