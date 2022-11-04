from __future__ import annotations
from typing import Union
import cv2
import PIL.ImageDraw
import PIL.ImageFont
import numpy as np


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
        od_resolution : resolution required for object detection model
        display_resolution : resolution for displayed video
        class_names : mapping be
    """
    def __init__(self, path: str, od_resolution: tuple[int, int], display_resolution: tuple[int, int]) -> None:
        self.filename = path
        self.od_resolution = od_resolution
        self.display_resolution = display_resolution

        self.class_names = {
            1: "person",
            2: "big_car",
            3: "car",
            4: "bike",
            5: "train",
            6: "traffic_light",
            7: "animal",
            8: "obstacle"
        }

    def __enter__(self) -> VideoReader:
        self.cap = cv2.VideoCapture(self.filename)
        if self.cap.isOpened() == False:
            return False
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.cap.release()
        cv2.destroyAllWindows()

    def get_frame(self) -> tuple[bool, np.ndarray]:
        """
        Method for retrieving frame from video file

            :return[0]: True if frame not valid, False if valid
            :return[1]: Frame data if valid, 0 if invalid
        """
        ret, self.frame = self.cap.read()

        if not ret:
            return True, 0

        return False, cv2.resize(self.frame, self.od_resolution)

    def set_frame(self, frame: np.ndarray) -> None:
        self.frame = frame

    def show_frame(self, annotated: bool = True) -> Union[bool, None]:
        """
        Method for displaying frame

            :param annotated: controls if displaying raw frame, or annotated with detection bounding boxes

            :return: True if user interrupt
        """
        if annotated:
            frame = self.annonated_frame
        else:
            frame = self.frame

        cv2.imshow('', cv2.resize(frame, self.display_resolution))

        if cv2.waitKey(1) == ord('q'):
            return True


    def annonate_image(self, boxes: np.ndarray, classes: np.ndarray, distances: np.ndarray) -> None:
        """
        Method for annotating images with bounding boxes

        :param boxes: bounding boxes indices from object detection model
        :param classes: classes ids from object detection model
        :param distances: estimated distances to objects
        """
        annonated_frame = PIL.Image.fromarray(cv2.resize(self.frame, self.od_resolution))
        draw = PIL.ImageDraw.Draw(annonated_frame)

        for i, box in enumerate(boxes):
            class_detected = classes[i]

            # Remember to adjust this
            box = (box[1], box[0], box[3], box[2])

            color = tuple((0, 0, 122))

            draw.rectangle(box, outline=color, width=2)

            distance = distances[i]
            if distance:
                text = f"{self.class_names[class_detected]} at {distance:.2f} m"
            else:
                text = self.class_names[class_detected]

            font = PIL.ImageFont.truetype("arial.ttf", 8)
            text_w, text_h = draw.textsize(text, font)
            draw.rectangle((box[0], box[1], box[0] + text_w, box[1] + text_h), fill=color, outline=color)
            draw.text((box[0], box[1]), text, fill=(0, 0, 0), font=font)
        
        self.annonated_frame = np.array(annonated_frame)