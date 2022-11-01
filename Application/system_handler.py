from models.midas.midas import MiDas
from models.dis_net.dis_net import DisNet
from models.object_detector.object_detector import ObjectDetector
from models.deep_sort.depsort import DeepSort
from models.model_loader import ModelLoader
from video_reader import VideoReader
import tensorflow as tf


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

    """
    def __init__(self, model_loader: ModelLoader) -> None:
        self.detector = ObjectDetector(model_loader.detection_model)
        self.od_resolution = model_loader.od_resolution
        self.disnet = DisNet(model_loader.distance_model)
        self.midas = MiDas(model_loader.depth_model)
        self.tracker = DeepSort()

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
                    frame = video.get_frame()

                    if frame[0]:  # break if no valid frame is retrieved
                        break

                    frame = frame[1]

                    # TODO - use wrapper here
                    img_detetect, detections = self.detector.predict(frame)

                    ind = detections['detection_scores'] > 0.6
                    boxes = detections['detection_boxes'][ind].numpy()
                    classes = detections['detection_classes'][ind].numpy()

                    boxes = boxes * 320
                    boxes = boxes.astype(int)

                    reader.annonate_image(boxes, classes)

                    if reader.show_frame():  # break on user interrupt
                        break

