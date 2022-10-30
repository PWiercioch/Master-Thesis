import cv2
import tensorflow as tf
import numpy as np
from typing import Union


class ObjectDetector:
    """
    This is a handler class for object detection model

        Parameters
        ----------
            detection_model : loaded tensorflow Object Detection API model

        Attributes
        ----------
            model : loaded tensorflow Object Detection API model
    """
    def __init__(self, detection_model) -> None:
        self.model = detection_model

    def __preprocess_image(self, img: np.ndarray[Union[int, float], Union[int, float], Union[int, float]]) \
            -> tuple[np.ndarray[Union[int, float], Union[int, float], Union[int, float]],
                     tf.python.framework.ops.EagerTensor]:
        """
        Prepares image for processing by creating a tensor

            :param img: image for object detection

            :return: input image and tensor for object detection model
        """
        img = cv2.resize(img, (320, 320)) #* 255
        # img = img.astype(np.uint8)
        input_tensor = tf.convert_to_tensor(img)

        # The model expects a batch of images, so add an axis with `tf.newaxis`.
        return img, input_tensor[tf.newaxis, ...]

    def predict(self, img: np.ndarray[Union[int, float], Union[int, float], Union[int, float]]) \
            -> tuple[np.ndarray[Union[int, float], Union[int, float], Union[int, float]], dict]:
        """
        Detect objects in an  image

            :param img: image for object detection

            :return:input image and detections
        """
        img, input_tensor = self.__preprocess_image(img)
        return img, self.model(input_tensor)