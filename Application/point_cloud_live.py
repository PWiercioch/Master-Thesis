import open3d as o3d
import numpy as np
import time
import pandas as pd
from point_cloud_base import PointCloudBase


class PointCloudLive(PointCloudBase):
    # def __init__(self, params):
    #     super(PointCloudBase, self).__init__(params)

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def set_imgs(self, rgb, depth):
        self.rgb_input = rgb
        self.depth_input = depth

    def _read_data(self):
        self.rgb_frame = self.rgb_input
        self.depth_frame = self.depth_input

        self.ret = True
