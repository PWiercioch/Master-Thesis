import open3d as o3d
import cv2
import numpy as np
import time


class PointCloudHandler:
    """

    """
    def __init__(self, rgb: str, depth: str) -> None:
        self.rgb_path = rgb
        self.depth_path = depth

        self.pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)

    def __open(self):
        self.cap1 = cv2.VideoCapture(self.rgb_path)
        self.cap2 = cv2.VideoCapture(self.depth_path)

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
        depth = np.invert(self.depth_frame[::, ::, 0])
        depth = depth /2 + 100
        o3d_rgb = o3d.geometry.Image(self.rgb_frame)
        o3d_a = o3d.geometry.Image(depth.astype(np.uint8))

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d_rgb, o3d_a)

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, self.pinhole_camera_intrinsic)

        return pcd

    def prepare_point_cloud(self):
        pcd = self.get_point_cloud()

        flip_transform = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
        pcd.transform(flip_transform)

        return pcd

    def change_viewport(self, v):
        control = v.get_view_control()
        # control.set_lookat([1, 1, 0])
        control.set_zoom(0.2)
        control.translate(100, 0, 0)
        v.register_animation_callback(self.update_view)

    def update_view(self, v):
        self.read_video()
        if self.ret:
            pcd = self.prepare_point_cloud()

            self.pcd.points = pcd.points
            self.pcd.colors = pcd.colors
            v.update_geometry(self.pcd)
            v.register_animation_callback(self.update_view)

        else:
            v.register_animation_callback(self.stop_animation)

    def stop_animation(self, v):
        v.destroy_window()
        self.__close()

    def show(self) -> None:
        """

        :return:
        """
        self.__open()

        # create visualization window
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=800, height=800)

        # create geometry
        geometry = o3d.geometry.PointCloud()
        vis.add_geometry(geometry)

        self.read_video()

        if self.ret:
            self.pcd = self.prepare_point_cloud()

            vis.register_animation_callback(self.change_viewport)

            vis.add_geometry(self.pcd)

            vis.run()

