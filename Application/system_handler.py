from models.midas.midas import MiDas
from models.dis_net.dis_net import DisNet
from models.object_detector.object_detector import ObjectDetector
from models.deep_sort.depsort import DeepSort
from models.model_loader import ModelLoader
from video_reader import VideoReader
import tensorflow as tf
import numpy as np
import cv2
from video_recorder import VideoRecorder
from wrappers.detection_wrapper import DetectionWrapper
from wrappers.distance_wrapper import DistanceWrapper
from wrappers.point_cloud_wrapper import PointCloudWrapper
from wrappers.depth_wrapper import DeothWrapper
from wrappers.writer_wrapper import WriterWrapper
from point_cloud_live import PointCloudLive
import threading

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
            "record_alpha_blended": True,
            "display_image": True,
            "return_depth": False
        }  # maybe provide a parameter or getter/setter

        self.od_threshold = 0.6  # maybe provide a parameter or getter/setter

    def weighted_focal(self, focal_h, focal_v, scores, distances):
        focal_h = focal_h[distances != None]
        focal_v = focal_v[distances != None]
        return np.mean(focal_h), np.mean(focal_v)
        focal_h = focal_h[distances != None]
        focal_v = focal_v[distances != None]
        scores = scores[distances != None]

        h_sum = []
        v_sum = []
        for h, v, s in zip(focal_h, focal_v, scores):
            h_sum.append(h * s)
            v_sum.append(v * s)

        return sum(h_sum) / scores.sum(), sum(v_sum) / scores.sum()

    def process_img(self, path: str, disp_res):
        img = cv2.imread(path, 1)
        img = cv2.resize(img, (self.od_resolution, self.od_resolution))

        reader = VideoReader(path, self.od_resolution, disp_res)
        reader.set_frame(img, "raw")

        ids, boxes, classes, scores = self._process_detections(img)

        depth_frame, inv_rel_depth = self._process_depth(reader, img)

        distances = self._process_distances(boxes, classes)

        focal_v, focal_h = self.calculate_focals(boxes, classes, distances)

        fit_status = self._process_regression(inv_rel_depth, boxes, distances)

        reader.annonate_image(reader.get_frame("raw"), boxes, classes, distances, ids, str("fit_status"))

        if self.config['display_image']:
            cv2.imshow('', reader.get_frame("annotated"))
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        if self.use_midas and self.use_disnet and fit_status:
            if self.distance_regressor.regression_model.get_coeffs()[1][0] < 0:
                intercept_state = False
            else:
                intercept_state = True

            focal_h, focal_v = self.weighted_focal(focal_h, focal_v, scores, distances)

            # print(classes)
            # print(scores)
            # print(focal_h, focal_v)

            focal_h = np.median(focal_h)
            focal_v = np.median(focal_v)

            # print(focal_h, focal_v)

            dimension = reader.get_frame("alpha_record").shape[0]
            center = dimension / 2
            # print(mean_focal, self.distance_regressor.regression_model.get_coeffs())
            pch = PointCloudLive([dimension, dimension, focal_h, focal_v, center, center],
                                 *self.distance_regressor.regression_model.get_coeffs())
            pch.set_imgs(img, reader.get_frame("alpha_record"))
            pch.show()

            if self.config["return_depth"]:
                return pch.pcd, reader.get_frame("alpha_record"), intercept_state

        if self.use_midas and self.use_disnet:
            return None, None, None

        return boxes, classes, distances

    def point_cloud_thread(self):
        self.pch = PointCloudLive()
        self.pch.show()

    def main_thread(self, reader):
        with reader as video:
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

                    fit_status = self._process_regression(inv_rel_depth, boxes, distances)

                    reader.annonate_image(video.get_frame("raw"), boxes, classes, distances, ids, "")

                    if fit_status:
                        focal_h, focal_v = self.weighted_focal(focal_h, focal_v, scores, distances)

                        focal_h = np.median(focal_h)
                        focal_v = np.median(focal_v)

                        dimension = reader.get_frame("alpha_record").shape[0]
                        center = dimension / 2
                        self.pch.set_camera_calib([dimension, dimension, focal_h, focal_v, center, center],
                                             *self.distance_regressor.regression_model.get_coeffs())
                        self.pch.set_imgs(frame, reader.get_frame("alpha_record"))
                        # self.pch.update_view_callback()
                        cv2.imshow('', reader.get_frame("annotated"))

                    key_pressed = cv2.waitKey(1)

                    if key_pressed:
                        cv2.waitKey(0)

                    if reader.show_frame():  # break on user interrupt
                        break

    def process_video(self, path: str, out_path: str, disp_res: int) -> None:
        """
        Main loop for processing input video: detects objects, annotates frames and displays them

            :param path: path to video file
            :param out_path: path for output video file
            :param disp_res: resolution for displayed video, assumed to be square
        """
        reader = VideoReader(path, self.od_resolution, disp_res)
        print("starting thread")
        pch_t = threading.Thread(target=self.point_cloud_thread)
        pch_t.start()
        print("thread started")
        main_t = threading.Thread(target=self.main_thread, args=[reader])
        main_t.start()

        main_t.join()


