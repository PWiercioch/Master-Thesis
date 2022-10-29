import numpy as np


class DisNet:
    def __init__(self, distance_model):
        self.model = distance_model
        self.class_sizes = {1:[175, 55, 30], 2: [110, 50, 180], 
            3: [160, 180, 400], 18: [50, 30, 60]}
        self.zoom_in_factor = 1

    def invert_dimensions(self, width, height, diagonal):
        inv_width = 1 / width if width !=0 else 0
        inv_height = 1 / height if height !=0 else 0
        inv_diagonal = 1 / diagonal if diagonal !=0 else 0

        return inv_width, inv_height, inv_diagonal

    def load_dist_input(self, predict_box, predict_class, img_width, img_height):
        top, left, bottom, right = predict_box
        width = float(right - left) / img_width
        height = float(bottom - top) / img_height
        diagonal = np.sqrt(np.square(width) + np.square(height))
        class_h, class_w, class_d = np.array(self.class_sizes[predict_class], dtype=np.float32)
        dist_input = [*self.invert_dimensions(width, height, diagonal), class_h, class_w, class_d]
        return np.array(dist_input)

    def predict(self, bounding_box, class_deteted):
        distance_input = self.load_dist_input(bounding_box, class_deteted, 320, 320)
        distance = self.model.predict(np.array([distance_input]).reshape(-1, 6)) * self.zoom_in_factor
        return distance

