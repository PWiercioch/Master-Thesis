from __future__ import annotations
from typing import Union
import cv2
import PIL.ImageDraw
import PIL.ImageFont
import numpy as np
import random
import time


class VideoReader:
    """
    Class for handling video input and display

        Parameters
        ----------
            path : path to video file
            od_resolution : resolution required for object detection model
            display_resolution : resolution for displayed video

        Attributes
        ----------
            filename : path to video file
            od_resolution : resolution required for object detection model, assumed to be square
            display_resolution : resolution for displayed video, assumed to be square
            scale : ratio between object detector required resolution and display resolution
            class_names : mapping be
            frame : read video frame
            frame_t : time of read frame
            cap : input video reader object
            frames :
    """
    def __init__(self, path: str, od_resolution: int, display_resolution: int) -> None:
        self.filename = path
        self.od_resolution = (od_resolution, od_resolution)
        self.display_resolution = (display_resolution, display_resolution)
        self.scale = display_resolution / od_resolution

        self.colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for i in range(50)]

        self.frame_t = time.time()

        self.class_names = {
            1: "person",
            3: "big_car",
            2: "car",
            4: "bike",
            5: "train",
            6: "traffic_light",
            7: "animal",
            8: "obstacle"
        }

        self.frames = {"raw": None, "annotated": None, "alpha_record": None}  # TODO - store frames at different stages

    def __enter__(self) -> VideoReader:
        self.cap = cv2.VideoCapture(self.filename)
        if not self.cap.isOpened():
            return False
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.cap.release()
        cv2.destroyAllWindows()

    def read_frame(self) -> tuple[bool, np.ndarray]:
        """
        Method for retrieving frame from video file

            :return[0]: True if frame not valid, False if valid
            :return[1]: Frame data if valid, 0 if invalid
        """
        ret, self.frames["raw"] = self.cap.read()

        self.frame_t = time.time()

        if not ret:
            return True, np.array([])

        return False, cv2.resize(self.frames["raw"], self.od_resolution)

    def set_frame(self, frame: np.ndarray, key: str) -> None:
        """

            :param frame:
            :param key:
            :return:
        """
        try:
            self.frames[key] = frame
        except KeyError:
            self.frames["raw"] = frame

    def get_frame(self, key: str) -> np.ndarray:
        """

            :param key:
            :return:
        """
        try:
            return self.frames[key]
        except KeyError:
            return self.frames[key]

    def show_frame(self, annotated: bool = True) -> Union[bool, None]:
        """
        Method for displaying frame

            :param annotated: controls if displaying raw frame, or annotated with detection bounding boxes

            :return: True if user interrupt
        """
        if annotated:  # TODO - not necessary - refactor
            frame = self.frames["annotated"]
        else:
            frame = self.frames["raw"]

        cv2.imshow('', cv2.resize(frame, self.display_resolution))

        if cv2.waitKey(1) == ord('q'):
            return True

    def annonate_image(self, frame: np.ndarray, boxes: np.ndarray, classes: np.ndarray, distances: np.ndarray,
                       ids: np.ndarray, comment: str = "") -> None:
        """
        Method for annotating images with bounding boxes

        :param frame:
        :param boxes: bounding boxes indices from object detection model
        :param classes: classes ids from object detection model
        :param distances: estimated distances to objects
        :param ids: list of tracked object ids
        :param comment:
        """
        annonated_frame = PIL.Image.fromarray(cv2.resize(frame, self.display_resolution))
        draw = PIL.ImageDraw.Draw(annonated_frame)

        font = PIL.ImageFont.truetype("arial.ttf", int(8 * self.scale))

        boxes = boxes * self.scale
        boxes = boxes.astype(int)

        for i, box in enumerate(boxes):
            class_detected = classes[i]

            # maybe move to system handler
            box = (box[1], box[0], box[3], box[2])

            color = self.colors[ids[i] % len(self.colors)]

            draw.rectangle(box, outline=color, width=int(2 * self.scale))

            distance = distances[i]
            if distance:
                text = f"{self.class_names[class_detected]} at {distance:.2f} m"
            else:
                text = self.class_names[class_detected]

            text_w, text_h = draw.textsize(text, font)
            draw.rectangle((box[0], box[1], box[0] + text_w, box[1] + text_h), fill=color, outline=color)
            draw.text((box[0], box[1]), text, fill=(0, 0, 0), font=font)

        draw.text((0, 0), f"{1 / (time.time() - self.frame_t):.3f} fps\n{comment}", fill=(0, 0, 255), font=font)

        self.frames["annotated"] = np.array(annonated_frame)
