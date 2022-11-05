from __future__ import annotations
import cv2
import numpy as np



class VideoCompare:
    """
    Class for comparing videos frame by frame

        Parameters
        ----------
            file1 : path to first video file
            file2 : path to second video file
            out_path : path for output video
            resolution : resolution for output video

        Attributes
        ----------
            resolution : resolution for output video
            half_resolution : resolution to which input frame should be resized to fit both frames in output
    """
    def __init__(self, file1, file2, out_path, resolution):
        self.resolution = resolution
        self.half_resolution = (int(resolution[1] / 2), resolution[0])
        self.cap1 = cv2.VideoCapture(file1)
        self.cap2 = cv2.VideoCapture(file2)
        self.out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, resolution)


    def __close(self) -> None:
        """
        Method for clean exit
        """
        self.cap1.release()
        self.cap2.release()
        self.out.release()

        cv2.destroyAllWindows()

    def __fill_frame(self, frame1, frame2):
        """

            :param frame1:
            :param frame2:

            :return:
        """
        frame1 = cv2.resize(frame1, self.half_resolution)
        frame2 = cv2.resize(frame2, self.half_resolution)
        result = np.zeros((*self.resolution, 3), np.uint8)
        result[:self.resolution[1], :self.half_resolution[0], ::] = frame1
        result[:self.resolution[1], self.half_resolution[0]:self.resolution[0], ::] = frame2
        return result

    def compare(self):
        """
        Method to compare videos
        """
        while (self.cap1.isOpened() and self.cap1.isOpened()):
            ret1, frame1 = self.cap1.read()
            ret2, frame2 = self.cap2.read()

            if ret1 and ret2:
                result = self.__fill_frame(frame1, frame2)

                self.out.write(result)

                cv2.imshow('', result)

                if cv2.waitKey(1) == ord('q'):
                    self.__close()
                    break

            else:
                self.__close()
                break

