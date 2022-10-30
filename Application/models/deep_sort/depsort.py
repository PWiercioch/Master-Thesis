from . import nn_matching
from .tracker import Tracker
from . import generate_detections as gdet


class DeepSort:
    """
    Wrapper for DeepSORT model

        Parameters
        ----------
            max_cosine_distance : maximal cosine distance for object assosiation
            max_age : number of frames after track will be deleted

        Attributes
        ----------
            encoder : used for extracting features for tracking algo
            tracker : tracking algo
    """
    def __init__(self, max_cosine_distance: float = 0.5, max_age: int = 5) -> None:
        max_cosine_distance = max_cosine_distance
        nn_budget = None

        # initialize deep sort object
        model_filename = r'models/deep_sort/model_data/mars-small128.pb'
        self.encoder = gdet.create_box_encoder(model_filename, batch_size=1)

        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        self.tracker = Tracker(metric, max_age=max_age)