import numpy as np
from abc import ABC, abstractmethod

class BaseRegressor(ABC):
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def get_coeffs(self):
        pass

    @abstractmethod
    def fit(self, regions, distances):
        pass

    @abstractmethod
    def predict(self, frame):
        pass

    @abstractmethod
    def predict_inverse(self, region):
        pass
