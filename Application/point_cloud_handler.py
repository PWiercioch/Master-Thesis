import open3d as o3d
import cv2
import numpy as np
import time
import pandas as pd


class PointCloudHandler:
    """

    """
    def __init__(self, rgb: str, depth: str, params: list[int, int, float, float, float, float]) -> None:
        self.rgb_path = rgb
        self.depth_path = depth
        log_path = f"{depth[:-4]}.pickle"
        self.log = pd.read_pickle(log_path)
        self.pause = False

        self.pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(*params)

    def __open(self):
        self.cap1 = cv2.VideoCapture(self.rgb_path)
        self.cap2 = cv2.VideoCapture(self.depth_path)
        self.frame_num = 0

    def __close(self) -> None:
        """
        Method for clean exit
        """
        self.cap1.release()
        self.cap2.release()

    def read_video(self):
        ret1, self.rgb_frame = self.cap1.read()
        ret2, self.depth_frame = self.cap2.read()

        self.ret = ret1 and ret2

    def get_point_cloud(self):
        """

        :param rgb:
        :param depth:
        :return:
        """
        depth = self.depth_frame[::, ::, 0] * 0.09476 + 5.5
        # This is in meters, technically should be converted to mm but it seems to wrok the same

        # convert bgr to rgb
        rgb = self.rgb_frame.copy()
        rgb[::, ::, 0] = self.rgb_frame[::, ::, 2]
        rgb[::, ::, 2] = self.rgb_frame[::, ::, 0]

        o3d_rgb = o3d.geometry.Image(rgb)
        o3d_a = o3d.geometry.Image(depth.astype(np.float32))

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d_rgb, o3d_a, convert_rgb_to_intensity=False)

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, self.pinhole_camera_intrinsic)

        return pcd

    def prepare_point_cloud(self):
        pcd = self.get_point_cloud()

        flip_transform = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
        pcd.transform(flip_transform)

        pcd = pcd.uniform_down_sample(every_k_points=22)

        return pcd

    def change_viewport(self, v):
        control = v.get_view_control()
        # control.set_lookat([1, 1, 0])
        control.set_zoom(0.5)
        control.translate(-50, 0, 0)
        v.register_animation_callback(self.update_view)

    def update_view(self, v):
        if self.pause:
            time.sleep(0.1)
            v.register_animation_callback(self.update_view)
        else:
            self.read_video()
            if self.ret:
                pcd = self.prepare_point_cloud()

                self.pcd.points = pcd.points
                self.pcd.colors = pcd.colors
                v.update_geometry(self.pcd)
                v.register_animation_callback(self.update_view)
                self.frame_num += 1

            else:
                v.register_animation_callback(self.stop_animation)

    def stop_animation(self, v):
        v.destroy_window()
        self.__close()

    def key_action_callback(self, vis, action, mods):
        if action == 1:  # key down
            if self.pause:
                self.pause = False
            else:
                self.pause = True

        return True

    def show(self) -> None:
        """

        :return:
        """
        self.__open()

        # create visualization window
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(width=800, height=800)

        # create geometry
        geometry = o3d.geometry.PointCloud()
        vis.add_geometry(geometry)
        # key_action_callback will be triggered when there's a keyboard press, release or repeat event
        vis.register_key_action_callback(32, self.key_action_callback)  # space

        self.read_video()

        if self.ret:
            self.pcd = self.prepare_point_cloud()

            vis.register_animation_callback(self.change_viewport)

            vis.add_geometry(self.pcd)

            vis.run()

