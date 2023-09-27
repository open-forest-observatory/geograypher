import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Tuple

import numpy as np
import numpy.ma as ma
import pyvista as pv
import torch
from pytorch3d.renderer import PerspectiveCameras
from pyvista import demos

from semantic_mesh_pytorch3d.config import PATH_TYPE


class MetashapeCamera:
    def __init__(
        self,
        image_filename: PATH_TYPE,
        transform: np.ndarray,
        f: float,
        cx: float,
        cy: float,
        image_width: int,
        image_height: int,
    ):
        """_summary_

        Args:
            image_filename (PATH_TYPE): _description_
            transform (np.ndarray): _description_
            f (float): _description_
            cx (float): _description_
            cy (float): _description_
            image_width (int): _description_
            image_height (int): _description_
        """
        self.image_filename = image_filename
        self.transform = transform
        self.f = f
        self.cx = cx
        self.cy = cy
        self.image_width = image_width
        self.image_height = image_height

    def rescale(self, scale: float):
        """_summary_

        Args:
            scale (float): _description_
        """
        self.transform[:3, 3] = self.transform[:3, 3] * scale

    def validate_projected_in_image(
        self, projected_verts: np.ndarray, image_size: Tuple[int, int]
    ):
        """_summary_

        Args:
            projected_verts (np.ndarray): _description_
            image_size (Tuple[int, int]): _description_

        Returns:
            _type_: _description_
        """
        image_space_verts = projected_verts[:2] / projected_verts[2:3]
        image_space_verts = image_space_verts.T

        in_front_of_cam = projected_verts[2] > 0

        img_width, image_height = image_size
        # Pytorch doesn't have a reduce operator, so this is the equivilent using boolean multiplication
        valid_verts_bool = (
            (image_space_verts[:, 0] > 0)
            * (image_space_verts[:, 1] > 0)
            * (image_space_verts[:, 0] < img_width)
            * (image_space_verts[:, 1] < image_height)
            * in_front_of_cam
        )
        valid_image_space_verts = image_space_verts[valid_verts_bool, :].to(torch.int)
        return valid_verts_bool.cpu().numpy(), valid_image_space_verts.cpu().numpy()

    def extract_colors(
        self, valid_bool: np.ndarray, valid_locs: np.ndarray, img: np.ndarray
    ):
        """_summary_

        Args:
            valid_bool (np.ndarray): _description_
            valid_locs (np.ndarray): _description_
            img (np.ndarray): _description_

        Returns:
            _type_: _description_
        """
        colors_per_vertex = np.zeros((valid_bool.shape[0], img.shape[2]))
        mask = np.ones((valid_bool.shape[0], img.shape[2])).astype(bool)
        valid_inds = np.where(valid_bool)[0]
        # Set the entries which are valid
        mask[valid_inds, :] = False

        i_locs = valid_locs[:, 1]
        j_locs = valid_locs[:, 0]
        valid_color_samples = img[i_locs, j_locs, :]
        colors_per_vertex[valid_inds, :] = valid_color_samples
        masked_color_per_vertex = ma.array(colors_per_vertex, mask=mask)
        return masked_color_per_vertex

    def splat_mesh_verts(self, mesh_verts: np.ndarray, img: np.ndarray, device: str):
        transform_4x4_world_to_cam = torch.Tensor(np.linalg.inv(self.transform)).to(
            device
        )
        """_summary_

        Returns:
            _type_: _description_
        """
        K = np.eye(3)
        K[0, 0] = self.f
        K[1, 1] = self.f
        K[0, 2] = self.cx + self.image_width / 2.0
        K[1, 2] = self.cy + self.image_height / 2.0
        K = torch.Tensor(K).to(device)
        homogenous_mesh_verts = torch.concatenate(
            (
                torch.Tensor(mesh_verts).to(device),
                torch.ones((mesh_verts.shape[0], 1)).to(device),
            ),
            axis=1,
        ).T
        camera_frame_mesh_verts = transform_4x4_world_to_cam @ homogenous_mesh_verts
        camera_frame_mesh_verts = camera_frame_mesh_verts[:3]
        # TODO review terminology
        projected_verts = K @ camera_frame_mesh_verts
        valid_bool, valid_locs = self.validate_projected_in_image(
            projected_verts=projected_verts,
            image_size=(self.image_width, self.image_height),
        )
        colors_per_vertex = self.extract_colors(valid_bool, valid_locs, img)
        return colors_per_vertex

    def get_pytorch3d_camera(self, device: str):
        """_summary_

        Args:
            device (str): _description_

        Returns:
            _type_: _description_
        """
        # Invert this because it's cam to world and we need world to cam
        transform_4x4_world_to_cam = np.linalg.inv(self.transform)
        rotation_about_z = np.array(
            [[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        )
        # Rotate about the Z axis because the NDC coordinates are defined X: left, Y: up and we use X: right, Y: down
        # See https://pytorch3d.org/docs/cameras
        transform_4x4_world_to_cam = rotation_about_z @ transform_4x4_world_to_cam

        R = torch.Tensor(np.expand_dims(transform_4x4_world_to_cam[:3, :3].T, axis=0))
        T = torch.Tensor(np.expand_dims(transform_4x4_world_to_cam[:3, 3], axis=0))

        # The image size is height, width which completely disreguards any other conventions they use...
        image_size = ((self.image_height, self.image_width),)
        fcl_screen = (self.f,)

        prc_points_screen = (
            (self.image_width / 2 + self.cx, self.image_height / 2 + self.cy),
        )
        cameras = PerspectiveCameras(
            R=R,
            T=T,
            focal_length=fcl_screen,
            principal_point=prc_points_screen,
            device=device,
            in_ndc=False,  # screen coords
            image_size=image_size,
        )
        return cameras

    def vis(self, plotter: pv.Plotter, frustum_scale: float = 0.5):
        """_summary_

        Args:
            plotter (pv.Plotter): _description_
            frustum_scale (float, optional): _description_. Defaults to 0.5.
        """
        scaled_halfwidth = self.image_width / (self.f * 2)
        scaled_halfheight = self.image_height / (self.f * 2)

        scaled_cx = self.cx / self.f
        scaled_cy = self.cx / self.f

        vertices = np.vstack(
            (
                np.array(
                    [
                        [0, 0, 0],
                        [
                            scaled_cx + scaled_halfwidth,
                            scaled_cy + scaled_halfheight,
                            1,
                        ],
                        [
                            scaled_cx + scaled_halfwidth,
                            scaled_cy - scaled_halfheight,
                            1,
                        ],
                        [
                            scaled_cx - scaled_halfwidth,
                            scaled_cy - scaled_halfheight,
                            1,
                        ],
                        [
                            scaled_cx - scaled_halfwidth,
                            scaled_cy + scaled_halfheight,
                            1,
                        ],
                    ]
                ).T
                * frustum_scale,
                np.ones((1, 5)),
            )
        )
        colors = np.array(
            [[0, 0, 0], [1, 0, 0], [0, 0, 1], [0, 0, 0], [0, 0, 0]]
        ).astype(float)

        projected_vertices = self.transform @ vertices
        rescaled_projected_vertices = projected_vertices[:3] / projected_vertices[3:]
        ## mesh faces
        faces = np.hstack(
            [
                [3, 0, 1, 2],
                [3, 0, 2, 3],
                [3, 0, 3, 4],
                [3, 0, 4, 1],
                [3, 1, 2, 3],
                [3, 3, 4, 1],
            ]  # square  # triangle  # triangle
        )
        frustum = pv.PolyData(rescaled_projected_vertices[:3].T, faces)
        frustum["RGB"] = colors
        frustum.triangulate()
        plotter.add_mesh(frustum, scalars="RGB", rgb=True)


class MetashapeCameraSet:
    def __init__(self, camera_file: PATH_TYPE, image_folder: PATH_TYPE):
        """_summary_

        Args:
            camera_file (PATH_TYPE): _description_
            image_folder (PATH_TYPE): _description_
        """
        self.parse_metashape_cam_file(camera_file=camera_file)

        self.filenames = [
            str(list(Path(image_folder).glob(filename + "*"))[0])
            for filename in self.filenames
        ]  # Assume there's only one file with that extension
        self.cameras = []

        for filename, transform in zip(self.filenames, self.transforms):
            new_camera = MetashapeCamera(
                filename,
                transform,
                self.f,
                self.cx,
                self.cy,
                self.image_width,
                self.image_height,
            )
            self.cameras.append(new_camera)

    def rescale(self, scale: float):
        """_summary_

        Args:
            scale (float): _description_
        """
        for camera in self.cameras:
            camera.rescale(scale)

    def vis(self, plotter: pv.Plotter, add_orientation_cube: bool = False):
        """_summary_

        Args:
            plotter (pv.Plotter): _description_
            add_orientation_cube (bool, optional): _description_. Defaults to False.
        """
        for camera in self.cameras:
            camera.vis(plotter)
        if add_orientation_cube:
            # TODO Consider adding to a freestanding vis module
            ocube = demos.orientation_cube()
            plotter.add_mesh(ocube["cube"], show_edges=True)
            plotter.add_mesh(ocube["x_p"], color="blue")
            plotter.add_mesh(ocube["x_n"], color="blue")
            plotter.add_mesh(ocube["y_p"], color="green")
            plotter.add_mesh(ocube["y_n"], color="green")
            plotter.add_mesh(ocube["z_p"], color="red")
            plotter.add_mesh(ocube["z_n"], color="red")

    def make_4x4_transform(
        self, rotation_str: str, translation_str: str, scale_str: str = "1"
    ):
        """_summary_

        Args:
            rotation_str (str): _description_
            translation_str (str): _description_
            scale_str (str, optional): _description_. Defaults to "1".

        Returns:
            _type_: _description_
        """
        rotation_np = np.fromstring(rotation_str, sep=" ")
        rotation_np = np.reshape(rotation_np, (3, 3))
        translation_np = np.fromstring(translation_str, sep=" ")
        scale = float(scale_str)
        transform = np.eye(4)
        transform[:3, :3] = rotation_np * scale
        transform[:3, 3] = translation_np
        return transform

    def parse_metashape_cam_file(self, camera_file: str):
        """_summary_

        Args:
            camera_file (str): _description_

        Raises:
            ValueError: _description_
        """
        # Load the xml file
        # Taken from here https://rowelldionicio.com/parsing-xml-with-python-minidom/
        tree = ET.parse(camera_file)
        root = tree.getroot()
        # first level
        chunk = root[0]
        # second level
        sensors = chunk[0]

        # sensors info
        sensor = sensors[0]
        self.image_width = int(sensor[0].get("width"))
        self.image_height = int(sensor[0].get("height"))

        if len(sensor) > 8:
            calibration = sensor[7]
            self.f = float(calibration[1].text)
            self.cx = float(calibration[2].text)
            self.cy = float(calibration[3].text)
            if None in (self.f, self.cx, self.cy):
                ValueError("Incomplete calibration provided")

            # Get potentially-empty dict of distortion parameters
            self.distortion_dict = {
                calibration[i].tag: float(calibration[i].text)
                for i in range(3, len(calibration))
            }

        else:
            raise ValueError("No calibration provided")

        # Get the transform relating the arbitrary local coordinate system
        # to the earth-centered earth-fixed EPGS:4978 system that is used as a reference by metashape
        transform = chunk[1][0][0]
        rotation = transform[0].text
        translation = transform[1].text
        scale = transform[2].text
        self.local_to_epgs_4978_transform = self.make_4x4_transform(
            rotation, translation, scale
        )

        cameras = chunk[2]

        self.filenames = []
        self.transforms = []
        for camera in cameras:
            if len(camera) < 5:
                # skipping unaligned camera
                continue
            self.filenames.append(camera.get("label"))
            self.transforms.append(np.fromstring(camera[0].text, sep=" ").reshape(4, 4))
