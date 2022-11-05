from region_extractor import RegionExtractor
import numpy as np


class MeanExtractor(RegionExtractor):
    def __describe(self, region):
        return region.mean()
