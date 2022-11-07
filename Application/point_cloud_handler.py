import open3d as o3d
import cv2
import numpy as np
import time


class PointCloudHandler:
    """

    """
    def __init__(self, rgb: str, depth: str) -> None:
        self.cap1 = cv2.VideoCapture(rgb)
        self.cap2 = cv2.VideoCapture(depth)

        self.pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)

    def __close(self) -> None:
        """
        Method for clean exit
        """
        self.cap1.release()
        self.cap2.release()

    def get_point_cloud(self, rgb, depth):
        """

        :param rgb:
        :param depth:
        :return:
        """
        depth = np.invert(depth)
        depth = depth /2 + 100
        o3d_rgb = o3d.geometry.Image(rgb)
        o3d_a = o3d.geometry.Image(depth.astype(np.uint8))

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d_rgb, o3d_a)

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, self.pinhole_camera_intrinsic)

        return pcd


    def show(self) -> None:
        """

        :return:
        """
        # create visualization window
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=800, height=800)

        # geometry is the point cloud used in your animaiton
        geometry = o3d.geometry.PointCloud()
        vis.add_geometry(geometry)

        def change_viewport(v):
            control = v.get_view_control()
            # control.set_lookat([1, 1, 0])
            control.set_zoom(0.2)
            control.translate(100, 0, 0)
            v.register_animation_callback(update_view)

        def update_view(v):
            ret1, rgb = self.cap1.read()
            ret2, depth = self.cap2.read()


            if ret1 and ret2:
                # v.clear_geometries()
                pcd1 = self.get_point_cloud(rgb, depth[::, ::, 0])

                flip_transform = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
                pcd1.transform(flip_transform)

                pcd.points = pcd1.points
                pcd.colors = pcd1.colors
                vis.update_geometry(pcd)
                # v.add_geometry(pcd)
                # v.update_renderer()
                v.register_animation_callback(update_view)

        # while (self.cap1.isOpened() and self.cap1.isOpened()):
        ret1, rgb = self.cap1.read()
        ret2, depth = self.cap2.read()

        if ret1 and ret2:
            pcd = self.get_point_cloud(rgb, depth[::, ::, 0])

            flip_transform = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
            pcd.transform(flip_transform)

            vis.register_animation_callback(change_viewport)

            vis.add_geometry(pcd)
            vis.poll_events()
            vis.update_renderer()

            vis.run()

        self.__close()




            # else:
            #     break

        vis.destroy_window()

