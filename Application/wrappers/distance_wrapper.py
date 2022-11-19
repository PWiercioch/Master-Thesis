import numpy as np


class DistanceWrapper:
    def _get_distances(self, boxes: np.ndarray, classes: np.ndarray) -> np.ndarray:
        """
        Method for estimating distances to detected objects, if class has reference size defined in self.disnet

            :param boxes: detected objects bounding boxes
            :param classes: detected objects classes

            :return: detected objects estimated distance,
                none if class doesn't have reference size defined in self.disnet
        """
        distances = []
        for i, box in enumerate(boxes):
            class_detected = int(classes[i])
            if class_detected in self.disnet.class_sizes.keys():
                distance = self.disnet.predict([box[1], box[0], box[3], box[2]], class_detected)
                distance = distance[0][0]
                distances.append(distance)
            else:
                distances.append(None)

        return np.array(distances)

    def _process_distances(self, boxes, classes):
        if self.use_disnet:
            distances = self._get_distances(boxes, classes)
        else:
            distances = np.array([None] * len(boxes))

        return distances