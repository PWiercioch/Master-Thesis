from models.midas.midas import MiDas
from models.dis_net.dis_net import DisNet
from models.object_detector.object_detector import ObjectDetector
from models.deep_sort.depsort import DeepSort
from models.model_loader import ModelLoader
from video_reader import VideoReader
import tensorflow as tf
import numpy as np
from video_recorder import VideoRecorder


class SystemHandler:
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

        self.od_threshold = 0.6  # maybe provide a parameter or getter/setter

    def __get_detections(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        Method for estimating inverse relative depth of a frame

            :param frame: frame for depth estimation

            :return: frame alpha blended with black background basing on depth value for each pixel
        """
        midas_frame = self.midas.predict(frame)

        a = (midas_frame - midas_frame.min())/(midas_frame.max() - midas_frame.min())
        blank = np.ones((320, 320, 3), np.uint8) * 255
        alpha = np.zeros((320, 320, 3), np.float64)

        alpha[::, ::, 0] = a
        alpha[::, ::, 1] = a
        alpha[::, ::, 2] = a

        frame = alpha * frame
        blank = (1.0 - alpha) * blank
        frame = frame + blank

        return frame.astype(np.uint8)

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

                    boxes, classes, scores = self.__get_detections(frame)

                    if self.use_deepsort:
                        ids, boxes, classes = self.tracker.predict(frame, boxes, classes, scores)
                    else:
                        ids = np.array([0] * len(boxes))

                    if self.use_midas:
                        depth_frame = self.__get_depth(frame)
                        reader.set_frame(depth_frame)

                    if self.use_disnet:
                        distances = self.__get_distances(boxes, classes)
                    else:
                        distances = np.array([None] * len(boxes))

                    reader.annonate_image(boxes, classes, distances, ids)

                    out.write(video.get_frame())

                    if reader.show_frame():  # break on user interrupt
                        break

