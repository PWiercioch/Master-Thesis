from models.midas.midas import MiDas
from models.dis_net.dis_net import DisNet
from models.object_detector.object_detector import ObjectDetector
from models.deep_sort.depsort import DeepSort
from models.model_loader import ModelLoader
from video_reader import VideoReader
import tensorflow as tf
import numpy as np
import cv2


class SystemHandler:
    """
    This is the most upper lever handler of the system.

        Parameters
        ----------
        model_loader : Object with models loaded from disk

        Attributes
        ----------
        detector : object detection model instance
        od_resolution : resolution required for object detection model
        disnet : distance estimation model instance
        midas : depth estimation model instance
        tracker : object tracker instance
        od_threshold : object detection probability threshold

    """
    def __init__(self, model_loader: ModelLoader) -> None:
        self.detector = ObjectDetector(model_loader.detection_model)
        self.od_resolution = model_loader.od_resolution
        self.disnet = DisNet(model_loader.distance_model)
        self.midas = MiDas(model_loader.depth_model)
        self.tracker = DeepSort()

        self.od_threshold = 0.6  # maybe provide a parameter

    def __get_detections(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Method for getting bounding boxes and object classes from frame, when detection probability is high enough

            :param frame: video frame for object detection

            :return: detected objects bounding boxes and classes
        """
        img_detetect, detections = self.detector.predict(frame)

        ind = detections['detection_scores'] > self.od_threshold
        boxes = detections['detection_boxes'][ind].numpy()
        classes = detections['detection_classes'][ind].numpy()

        boxes = boxes * self.od_resolution[0]
        boxes = boxes.astype(int)

        return boxes, classes

    def __get_distances(self, boxes: np.ndarray, classes: np.ndarray) -> np.ndarray:
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

    def __get_depth(self, frame: np.ndarray) -> np.ndarray:
        """
        Method for estimating depth of a frame

            :param frame: frame for depth estimation

            :return: depth frame
        """
        frame = self.midas.predict(frame)
        frame = cv2.cvtColor(frame,cv2.COLOR_GRAY2RGB)
        return frame

    def process_video(self, path: str, disp_res: tuple[int, int]) -> None:
        """
        Main loop for processing input video: detects objects, annotates frames and displays them

        :param path: path to video file
        :param disp_res: resolution for displayed video
        """
        reader = VideoReader(path, self.od_resolution, disp_res)

        with reader as video:
            if not video:  # break if error while opening file
                return None

            with tf.device("/device:GPU:0"):

                while True:
                    ret, frame = video.get_frame()

                    if ret:  # break if no valid frame is retrieved
                        break

                    depth_frame = self.__get_depth(frame)
                    reader.set_frame(depth_frame)

                    boxes, classes = self.__get_detections(frame)
                    distances = self.__get_distances(boxes, classes)

                    reader.annonate_image(boxes, classes, distances)

                    if reader.show_frame():  # break on user interrupt
                        break

