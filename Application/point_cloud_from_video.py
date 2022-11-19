import open3d as o3d
import cv2
import numpy as np
import time
import pandas as pd
from point_cloud_base import PointCloudBase


class PointCloudFromVideo(PointCloudBase):
    """

    """
    def __init__(self, rgb: str, depth: str, params: list[int, int, float, float, float, float]) -> None:
        super().__init__(params)
        self.rgb_path = rgb
        self.depth_path = depth
        log_path = f"{depth[:-4]}.pickle"
        self.log = pd.read_pickle(log_path)

    def __enter__(self):
        self.cap1 = cv2.VideoCapture(self.rgb_path)
        self.cap2 = cv2.VideoCapture(self.depth_path)
        self.frame_num = 0

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        Method for clean exit
        """
        self.cap1.release()
        self.cap2.release()

    def _PointCloudBase__read_data(self):
        ret1, self.rgb_frame = self.cap1.read()
        ret2, self.depth_frame = self.cap2.read()

        self.ret = ret1 and ret2

    def update_view_callback(self, v):
        super().update_view_callback(v)
        self.frame_num += 1




