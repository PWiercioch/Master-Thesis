from .extractors.mean_extractor import MeanExtractor
from .extractors.center_extractor import CenterExtractor
from .regressors.linear_regressor import LinearRegressor
from .extractors.median_extractor import MedianExtractor
import numpy as np


class DistanceRegressor:
    """
    Method for acquiring absolute distance from relative inverse depth and distance measurements

    """
    def __init__(self, extractor_type, regressor_type, **kwargs):
        super().__init__()
        self.extractor = self.__get_extractor(extractor_type)(**kwargs)
        self.regression_model = self.__get_regressor(regressor_type)(**kwargs)

    def __get_extractor(self, extractor_type):
        extractors = {'mean': MeanExtractor, "center": CenterExtractor, "median": MedianExtractor}
        try:
            return extractors[extractor_type]
        except KeyError:
            print("Wrong key for region extractor type \n Choosing MeanExtractor")
            return MeanExtractor

    def __get_regressor(self, regressor_type):
        regressors = {"linear": LinearRegressor}
        try:
            return regressors[regressor_type]
        except KeyError:
            print("Wrong key for regressor type \n Choosing LinearRegressor")
            return LinearRegressor

    def predict(self, alpha, boxes, distances):
        regions = self.extractor.extract_regions(alpha, boxes)

        fitted = self.regression_model.fit(regions, distances)

        if fitted:
            return True, self.regression_model.predict(alpha)
        else:
            return False, np.empty([0, 0])
