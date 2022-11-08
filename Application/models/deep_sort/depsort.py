from . import nn_matching
from .tracker import Tracker
from .detection import Detection
from . import generate_detections as gdet
import numpy as np


# TODO - maybe store object distances as dict (needed to be cleared every frame)
class DeepSort:
    """
    Wrapper for DeepSORT model

        Parameters
        ----------
            max_cosine_distance : maximal cosine distance for object association
            max_age : number of frames after track will be deleted

        Attributes
        ----------
            encoder : used for extracting features for tracking algo
            tracker : tracking algo
    """
    def __init__(self, max_cosine_distance: float, max_age: int) -> None:
        max_cosine_distance = max_cosine_distance
        nn_budget = None

        # initialize deep sort object
        model_filename = r'models/deep_sort/model_data/mars-small128.pb'
        self.encoder = gdet.create_box_encoder(model_filename, batch_size=1)

        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        self.tracker = Tracker(metric, max_age=max_age)

    def __preprocess(self, frame, boxes, classes, scores) -> list[Detection]:
        """
        Method for updating tracked objects

            :param classes: list of detected objects classes
            :param frame: current video frame
            :param boxes: list of detected objects bounding boxes
            :param scores: list of detected objects probability scores

            :return: list of features for tracker
        """
        boxes = np.array(boxes)
        # TODO -check if width and height are correctly placed 
        boxes = [[box[0], box[1], abs(box[2] - box[0]), abs(box[3] - box[1])] for box in boxes]
        names = np.array(classes)
        scores = np.array(scores) + 0.2
        features = np.array(self.encoder(frame, boxes))
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in
                      zip(boxes, scores, names, features)]

        return detections

    def __postprocess(self) -> tuple[np.array, np.array, np.array]:
        """
        Method for getting current tracked objects info

            :return[0]: tracked objects ids
            :return[1]: tracked objects bounding boxes
            :return[2]: tracked objects classes
        """

        tracked_bboxes = []
        tracked_ids = []
        tracked_classes = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr().astype(np.int32)  # Get the corrected/predicted bounding box
            class_name = track.get_class()  # Get the class name of particular object
            tracking_id = track.track_id  # Get the ID for the particular track
            tracked_bboxes.append(bbox.tolist())  # Structure data, that we could use it with our draw_bbox function
            tracked_ids.append(tracking_id)
            tracked_classes.append(class_name)

        return np.array(tracked_ids), np.array(tracked_bboxes), np.array(tracked_classes)

    def predict(self, frame: np.ndarray, boxes: np.ndarray, classes: np.ndarray, scores: np.ndarray)\
            -> tuple[np.array, np.array, np.array]:
        """
        Method for updating tracker and getting tracked objects info

            :param classes: list of detected objects classes
            :param frame: current video frame
            :param boxes: list of detected objects bounding boxes
            :param scores: list of detected objects probability scores

            :return[0]: tracked objects ids
            :return[1]: tracked objects bounding boxes
            :return[2]: tracked objects classes
        """
        detections = self.__preprocess(frame, boxes, classes, scores)
        self.tracker.predict()
        self.tracker.update(detections)

        ids, boxes, classes = self.__postprocess()

        return ids, boxes, classes