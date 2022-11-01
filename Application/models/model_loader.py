import tensorflow as tf
import tensorflow_hub as hub


class ModelLoader:
    """
    Wrapper for loading models from disk. Allows to modify rest of the codebase without reloading models in notebook mode.

        Parameters
        ---------
            od_model_path : path to object detection model
            dis_model_path : path to distance estimation model
            midas_path : path to depth estimation model

        Attributes
        ----------
            detection_model : object detection model instance
            distance_model : distance estimation model instance
            depth_model : depth estimation model instance
    """
    def __init__(self, od_model_path: str, dis_model_path: str, midas_path: str) -> None:
        self.load_detection_model(od_model_path)
        self.load_distance_model(dis_model_path)
        self.load_depth_model(midas_path)

    def load_detection_model(self, path: str) -> None:
        self.detection_model = tf.saved_model.load(path)

    def load_distance_model(self, path: str) -> None:
        self.distance_model = tf.keras.models.load_model(path)

    def load_depth_model(self, path: str) -> None:
        self.depth_model = hub.load(path, tags=['serve'])