import numpy as np


class DetectionWrapper:
    def _get_detections(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Method for getting bounding boxes and object classes from frame, when detection probability is high enough

            :param frame: video frame for object detection

            :return: detected objects bounding boxes, classes and scores
        """
        img_detetect, detections = self.detector.predict(frame)

        ind = detections['detection_scores'] > self.od_threshold
        scores = detections['detection_scores'][ind].numpy()
        boxes = detections['detection_boxes'][ind].numpy()
        classes = detections['detection_classes'][ind].numpy()

        boxes = boxes * self.od_resolution
        boxes = boxes.astype(int)

        return boxes, classes, scores

    def _process_detections(self, frame):
        boxes, classes, scores = self._get_detections(frame)

        if self.use_deepsort:
            ids, boxes, classes = self.tracker.predict(frame, boxes, classes, scores)
        else:
            ids = np.array([0] * len(boxes))

        return ids, boxes, classes, scores