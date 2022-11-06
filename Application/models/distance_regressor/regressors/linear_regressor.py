import numpy as np
from sklearn.linear_model import LinearRegression
from .base_regressor import BaseRegressor


class LinearRegressor(BaseRegressor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = LinearRegression()
        self.inverse_model = LinearRegression()
        self.initialized = False  # TODO could allow to make predictions for model fitted with past frames (with max age)
        # TODO - also probably could use information about past tracked objects

    def fit(self, regions, distances):
        # TODO move data cleaning to a parent class
        regions = regions[regions != None]
        distances = distances[distances != None]
        if len(distances) >= 2:
            self.model.fit(regions.reshape(-1, 1), distances.reshape(-1, 1))
            self.inverse_model.fit(distances.reshape(-1, 1), regions.reshape(-1, 1))
            self.initialized = True
            return True
        else:
            return False

    def predict(self, frame):
        return self.model.coef_ * frame + self.model.intercept_

    def predict_inverse(self, region):
        return self.inverse_model.coef_ * region + self.inverse_model.intercept_
