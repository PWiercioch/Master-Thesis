from models.midas.midas import MiDas
from models.dis_net.dis_net import DisNet
from models.object_detector.object_detector import ObjectDetector
from models.deep_sort.depsort import DeepSort
from models.model_loader import ModelLoader
from video_reader import VideoReader
import tensorflow as tf
import numpy as np
from video_recorder import VideoRecorder
from wrappers.detection_wrapper import DetectionWrapper
from wrappers.distance_wrapper import DistanceWrapper
from wrappers.point_cloud_wrapper import PointCloudWrapper
from wrappers.depth_wrapper import DeothWrapper
from wrappers.writer_wrapper import WriterWrapper


class SystemHandler(DetectionWrapper, DistanceWrapper, DeothWrapper, PointCloudWrapper, WriterWrapper):
    """
    This is the most upper lever handler of the system.

        Parameters
        ----------
            model_loader : Object with models loaded from disk
            max_cosine_distance : maximal cosine distance for object association
            max_age : number of frames after track will be deleted

        Attributes
        ----------
            detector : object detection model instance
            od_resolution : resolution required for object detection model, assumed to be square
            disnet : distance estimation model instance
            midas : inverse relative depth estimation model instance
            tracker : object tracker instance
            distance_regressor : object for distance regression

            use_midas : estimate depth on an image or not
            use_disnet : estimate distance of objects or not
            use_deepsort : track objects ot not

            record_annotated :
            alpha_blending :

            od_threshold : object detection probability threshold

    """
    def __init__(self, model_loader: ModelLoader, max_cosine_distance: float = 0.5, max_age: int = 5) -> None:
        self.detector = ObjectDetector(model_loader.detection_model)
        self.od_resolution = model_loader.od_resolution
        self.disnet = DisNet(model_loader.distance_model)
        self.midas = MiDas(model_loader.depth_model)
        self.tracker = DeepSort(max_cosine_distance, max_age)
        self.distance_regressor = model_loader.distance_regressor

        self.use_midas = True  # maybe provide a parameter or getter/setter
        self.use_disnet = True  # maybe provide a parameter or getter/setter
        self.use_deepsort = True  # maybe provide a parameter or getter/setter

        self.config = {
            "record_annotated": True,
            "record_alpha_blended": True
        }  # maybe provide a parameter or getter/setter

        self.od_threshold = 0.6  # maybe provide a parameter or getter/setter

    def process_img(self, path: str):
        pass

    def process_video(self, path: str, out_path: str, disp_res: int) -> None:
        """
        Main loop for processing input video: detects objects, annotates frames and displays them

            :param path: path to video file
            :param out_path: path for output video file
            :param disp_res: resolution for displayed video, assumed to be square
        """
        reader = VideoReader(path, self.od_resolution, disp_res)
        writer = VideoRecorder(out_path, disp_res)

        with reader as video, writer as out:
            if not video:  # break if error while opening file
                return None

            with tf.device("/device:GPU:0"):

                while True:
                    ret, frame = video.read_frame()

                    if ret:  # break if no valid frame is retrieved
                        break

                    ids, boxes, classes, scores = self._process_detections(frame)

                    depth_frame, inv_rel_depth = self._process_depth(reader, frame)

                    distances = self._process_distances(boxes, classes)

                    focal_v, focal_h = self.calculate_focals(boxes, classes, distances)

                    fit_status = self._process_regression(inv_rel_depth,  boxes, distances)

                    reader.annonate_image(video.get_frame("raw"), boxes, classes, distances, ids, "")

                    self._write(out, video, fit_status, boxes, classes, scores, distances, focal_v, focal_h, "")

                    if reader.show_frame():  # break on user interrupt
                        break

