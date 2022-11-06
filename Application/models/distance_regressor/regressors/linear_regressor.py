import numpy as np
from sklearn.linear_model import LinearRegression
from .base_regressor import BaseRegressor


class LinearRegressor(BaseRegressor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = LinearRegression()
        self.inverse_model = LinearRegression()

    def fit(self, regions, distances):
        self.model.fit(regions, distances)
        self.inverse_model.fit(distances, regions)

    def predict(self, frame):
        return self.model.coef_ * frame + self.model.intercept_

    def predict_inverse(self, region):
        return self.inverse_model.coef_ * region + self.inverse_model.intercept_
