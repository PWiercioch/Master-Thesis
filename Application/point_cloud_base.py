from abc import ABC, abstractmethod
import open3d as o3d
import numpy as np
import time
import pandas as pd



class PointCloudBase(ABC):
    def __init__(self):
        self.pause = False
        self.ret = False


    @abstractmethod
    def _read_data(self):
        pass

    @abstractmethod
    def __enter__(self):
        pass

    @abstractmethod
    def __exit__(self):
        pass

    def set_camera_calib(self, params, coe, intercept):
        self.coe = coe[0][0]
        self.intercept = intercept[0]

        self.pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(*params)

    def _get_point_cloud(self):
        """

        :param rgb:
        :param depth:
        :return:
        """
        depth = self.depth_frame[::, ::, 0] * self.coe + self.intercept
        # This is in meters, technically should be converted to mm but it seems to wrok the same

        # convert bgr to rgb
        rgb = self.rgb_frame.copy()
        rgb[::, ::, 0] = self.rgb_frame[::, ::, 2]
        rgb[::, ::, 2] = self.rgb_frame[::, ::, 0]

        o3d_rgb = o3d.geometry.Image(rgb)
        o3d_a = o3d.geometry.Image(depth.astype(np.float32))

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d_rgb, o3d_a, convert_rgb_to_intensity=False, depth_scale=1.0, depth_trunc=1000.0)

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, self.pinhole_camera_intrinsic)

        return pcd

    def _prepare_point_cloud(self):
        pcd = self._get_point_cloud()

        flip_transform = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
        pcd.transform(flip_transform)

        # pcd = pcd.uniform_down_sample(every_k_points=2)

        return pcd

    def set_viewport_callback(self, v):
        control = v.get_view_control()
        control.set_zoom(0.5)
        control.translate(-50, 0, 0)

        #v.register_animation_callback(self.update_view) TODO execute at the start

    def update_view_callback(self, v):
        if self.pause:
            time.sleep(0.1)
            v.register_animation_callback(self.update_view_callback)
        else:
            self._read_data()
            if self.ret:
                pcd = self._prepare_point_cloud()

                self.pcd.points = pcd.points
                self.pcd.colors = pcd.colors
                v.update_geometry(self.pcd)
                v.register_animation_callback(self.update_view_callback)
                # self.frame_num += 1 TODO only in video reader

            # else:
            #     v.register_animation_callback(self.stop_animation)

    def stop_animation(self, v):
        v.destroy_window()


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
        # create visualization window
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(width=800, height=800)

        # create geometry
        geometry = o3d.geometry.PointCloud()
        vis.add_geometry(geometry)

        # key_action_callback will be triggered when there's a keyboard press, release or repeat event
        vis.register_key_action_callback(32, self.key_action_callback)  # space

        # self.__read_data()

        self.pcd = geometry

        if self.ret:
            # self.pcd = self.__prepare_point_cloud()

            vis.register_animation_callback(self.set_viewport_callback)
            vis.register_animation_callback(self.update_view_callback)

            # vis.add_geometry(self.pcd)

            vis.run()


