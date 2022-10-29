import tensorflow as tf
import cv2


class MiDas:
    def __init__(self, module):
        self.model = module


    def preprocess(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0

        img_resized = tf.image.resize(img, [256,256], method='bicubic', preserve_aspect_ratio=False)
        img_resized = tf.transpose(img_resized, [2, 0, 1])
        img_input = img_resized.numpy()
        reshape_img = img_input.reshape(1,3,256,256)
        tensor = tf.convert_to_tensor(reshape_img, dtype=tf.float32)

        return tensor

    def post_process(self, output, img):
        prediction = output['default'].numpy()
        prediction = prediction.reshape(256, 256)
        prediction = cv2.resize(prediction, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
        depth_min = prediction.min()
        depth_max = prediction.max()
        img_out = (255 * (prediction - depth_min) / (depth_max - depth_min)).astype("uint8")

        return img_out


    def predict(self, img):
        tensor = self.preprocess(img)
        output = self.model.signatures['serving_default'](tensor)
        result = self.post_process(output, img)

        return result