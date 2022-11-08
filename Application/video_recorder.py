from __future__ import annotations
import cv2
import numpy as np
import pandas as pd

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
        self.out = cv2.VideoWriter(f"{self.filename}.mp4", cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, self.resolution)

        self.boxes = []
        self.classes = []
        self.scores = []
        self.distances = []

        self.focal_v = []
        self.focal_h = []
        self.weights = []
        self.calibrated = []

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.out.release()
        df = pd.DataFrame({"boxes": self.boxes, "classes": self.classes, "scores": self.scores,
                           "distances": self.distances, "focal_vertical": self.focal_v,
                           "focal_horizontal": self.focal_h, "weights": self.weights, "calibrated": self.calibrated})

        df.to_pickle(f"{self.filename}.pickle")

    def log(self, boxes, classes, scores, distances, focal_v, focal_h, weights, calibrated):
        self.boxes.append(boxes)
        self.classes.append(classes)
        self.scores.append(scores)
        self.distances.append(distances)

        self.focal_v.append(focal_v)
        self.focal_h.append(focal_h)
        self.weights.append(weights)
        self.calibrated.append(calibrated)

    def write(self, frame: np.ndarray) -> None:
        self.out.write(cv2.resize(frame, self.resolution))
