import numpy as np
import tensorflow as tf
import tensorflow.python as tf_python
import cv2
from typing import Union


class MiDas:
    """
    This is a handler class for depth estimation model

        Parameters
        ----------
            model : loaded tensorflow hub depth estimation model

        Attributes
        ----------
            model : loaded tensorflow hub depth estimation model
    """
    def __init__(self, model: tf_python.trackable.autotrackable.AutoTrackable) -> None:
        self.model = model

    def __preprocess(self, img: np.ndarray) -> tf_python.framework.ops.EagerTensor:
        """
        Prepares image for processing by resizing and creating a tensor

            :param img: image for depth estimation

            :return: tensor ready to be processed by depth estimation model
        """
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0

        img_resized = tf.image.resize(img, [256,256], method='bicubic', preserve_aspect_ratio=False)
        img_resized = tf.transpose(img_resized, [2, 0, 1])
        img_input = img_resized.numpy()
        reshape_img = img_input.reshape(1,3,256,256)
        tensor = tf.convert_to_tensor(reshape_img, dtype=tf.float32)

        return tensor

    def __post_process(self, output: dict, img: np.ndarray) -> np.ndarray:
        """
        Process depth estimation output to get depth image

            :param output: output from depth estimation model
            :param img: image for depth estimation

            :return: Depth image
        """
        prediction = output['default'].numpy()
        prediction = prediction.reshape(256, 256)
        prediction = cv2.resize(prediction, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
        depth_min = prediction.min()
        depth_max = prediction.max()
        img_out = (255 * (prediction - depth_min) / (depth_max - depth_min)).astype("uint8")

        return img_out


    def predict(self, img: np.ndarray) -> np.ndarray:
        """
        Estiamte depth in an image

            :param img: image for depth estimation

            :return: estimation image depth
        """
        tensor = self.__preprocess(img)
        output = self.model.signatures['serving_default'](tensor)
        result = self.__post_process(output, img)

        return result