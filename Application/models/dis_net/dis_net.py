import numpy as np
import keras


class DisNet:
    """
    This is a handler class for distance estimation model

        Parameters
        ----------
            distance_model : loaded keras distance estimation model

        Attributes
        ----------
            model : loaded keras distance estimation model
            class_sizes : reference average class dimensions in centimeters
            zoom_in_factor : zoom factor, 1 for no zoom
    """
    def __init__(self, distance_model: keras.engine.sequential.Sequential) -> None:
        self.model = distance_model
        self.class_sizes = {1:[175, 55, 30], 2: [110, 50, 180], 
            3: [160, 180, 400], 18: [50, 30, 60]}
        self.zoom_in_factor = 1

    def __invert_dimensions(self, width: float, height: float, diagonal: float) -> tuple[float, float, float]:
        """
        Returns inverse dimensions, handles division by 0
        """
        inv_width = 1 / width if width !=0 else 0
        inv_height = 1 / height if height !=0 else 0
        inv_diagonal = 1 / diagonal if diagonal !=0 else 0

        return inv_width, inv_height, inv_diagonal

    def __load_dist_input(self, predict_box: list[float, float, float, float], predict_class: int,
                        img_width: int, img_height: int) -> np.ndarray:
        """
        Prepares input for distance estimation model from detection bounding box and class

            :param predict_box: bounding box coordinates
            :param predict_class: predicted object class
            :param img_width: width of image
            :param img_height: height of image

            :return: features for distance estimation model
        """
        top, left, bottom, right = predict_box
        width = float(right - left) / img_width
        height = float(bottom - top) / img_height
        diagonal = np.sqrt(np.square(width) + np.square(height))
        class_h, class_w, class_d = np.array(self.class_sizes[predict_class], dtype=np.float32)
        dist_input = [*self.__invert_dimensions(width, height, diagonal), class_h, class_w, class_d]
        return np.array(dist_input)

    def predict(self, bounding_box: list[float, float, float, float], class_deteted: int) -> float:
        """
        Predicts detected object distance

            :param bounding_box: bounding box coordinates
            :param class_deteted: predicted object class

            :return: predicted distance
        """
        distance_input = self.__load_dist_input(bounding_box, class_deteted, 320, 320)
        distance = self.model.predict(np.array([distance_input]).reshape(-1, 6)) * self.zoom_in_factor
        return distance

