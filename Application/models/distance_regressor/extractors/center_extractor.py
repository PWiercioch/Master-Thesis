from .region_extractor import RegionExtractor
import numpy as np


class CenterExtractor(RegionExtractor):
    def _RegionExtractor__describe(self, region):
        center = np.array(region.shape[0:2]) / 2
        center = center.astype(int)

        return region[center[0], center[1], 0]