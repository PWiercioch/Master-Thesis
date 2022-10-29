import cv2
import tensorflow as tf


class ObjectDetector:
    def __init__(self, detection_model):
        self.model = detection_model

    def preprocess_image(self, img):
        img = cv2.resize(img, (320, 320)) #* 255
        # img = img.astype(np.uint8)
        input_tensor = tf.convert_to_tensor(img)

        # The model expects a batch of images, so add an axis with `tf.newaxis`.
        return img, input_tensor[tf.newaxis, ...]

    def predict(self, img):
        img, input_tensor = self.preprocess_image(img)
        return img, self.model(input_tensor)