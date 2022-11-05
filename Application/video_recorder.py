from __future__ import annotations
import cv2
import numpy as np

class VideoRecorder:
    """
    Handler for recording data

        Parameters
        ----------
        filename : name of the video file to be saved
        resolution : resolution of recorded video

        Attributes
        ----------
        filename : name of the video file to be saved
        resolution : resolution of recorded video
        out : output video writer object

    """

    def __init__(self, filename, resolution):
        self.filename = filename
        self.resolution = (resolution, resolution)

    def __enter__(self) -> VideoRecorder:
        self.out = cv2.VideoWriter(self.filename, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, self.resolution)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.out.release()

    def write(self, frame: np.ndarray) -> None:
        self.out.write(cv2.resize(frame, self.resolution))
