from .region_extractor import RegionExtractor
import numpy as np


class MeanExtractor(RegionExtractor):
    def _RegionExtractor__describe(self, region):
        return region.mean()
