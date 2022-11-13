import numpy as np


class DeothWrapper:
    def _get_depth(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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

        # Only for recording purposes
        alpha = alpha * 255
        alpha = alpha.astype(np.uint8)
        alpha = np.invert(alpha)

        return frame.astype(np.uint8), alpha.astype(np.uint8)


    def _process_depth(self, reader, frame):
        if self.use_midas:
            depth_frame, inv_rel_depth = self._get_depth(frame)
            reader.set_frame(depth_frame, "raw")
            reader.set_frame(inv_rel_depth, "alpha_record")

            return depth_frame, inv_rel_depth
        else:
            return np.empty([]), np.empty([])
