import numpy as np
from abc import ABC, abstractmethod


class RegionExtractor(ABC):
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def __describe(self, region):
        pass

    def extract_regions(self, alpha, boxes):
        regions = []
        for box in boxes:
            region = alpha[box[0]:box[2], box[1]:box[3]]
            regions.append(self.__describe(region))

        return np.array(regions)
