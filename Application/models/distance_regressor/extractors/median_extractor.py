from .region_extractor import RegionExtractor
import numpy as np


class MedianExtractor(RegionExtractor):
    def _RegionExtractor__describe(self, region):
        return np.median(region)