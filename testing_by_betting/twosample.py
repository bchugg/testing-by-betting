from .base import AbstractBettor

class SingleDimensionONS(AbstractBettor): 

    def __init__(self, alpha=0.05) -> None:
        super().__init__(alpha, strategy='ONS')

    def predict(self, X1, X2):
        return X1 - X2