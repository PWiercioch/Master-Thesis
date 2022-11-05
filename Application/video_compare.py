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
            frame_by_frame : continuous playback or frame by frame

        Attributes
        ----------
            resolution : resolution for output video
            half_resolution : resolution to which input frame should be resized to fit both frames in output
            cap1 : first input video reader object
            cap2 : second input video reader object
            out : output video writer object
            type : 1 if continuous playback, 0 for frame-by-frame playback (on user input)
    """
    def __init__(self, file1: str, file2: str, out_path: str, resolution: tuple[int, int], frame_by_frame: bool = False):
        self.resolution = resolution
        self.half_resolution = (int(resolution[1] / 2), resolution[0])
        self.cap1 = cv2.VideoCapture(file1)
        self.cap2 = cv2.VideoCapture(file2)
        self.out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, resolution)

        if frame_by_frame:
            self.type = 0
        else:
            self.type = 1


    def __close(self) -> None:
        """
        Method for clean exit
        """
        self.cap1.release()
        self.cap2.release()
        self.out.release()

        cv2.destroyAllWindows()

    def __fill_frame(self, frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
        """
        Method to combine two compared frames into one frame

            :param frame1: frame from first video
            :param frame2: frame from second video

            :return: output frame with side by side frames 1 and 2
        """
        frame1 = cv2.resize(frame1, self.half_resolution)
        frame2 = cv2.resize(frame2, self.half_resolution)
        result = np.zeros((*self.resolution, 3), np.uint8)
        result[:self.resolution[1], :self.half_resolution[0], ::] = frame1
        result[:self.resolution[1], self.half_resolution[0]:self.resolution[0], ::] = frame2
        return result

    def compare(self) -> None:
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

                if cv2.waitKey(self.type) == ord('q'):
                    self.__close()
                    break

            else:
                self.__close()
                break

