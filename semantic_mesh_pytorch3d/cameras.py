import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Tuple

import numpy as np
import numpy.ma as ma
import pyvista as pv
import torch
from pytorch3d.renderer import PerspectiveCameras
from pyvista import demos
from skimage.io import imread
from skimage.transform import resize

from semantic_mesh_pytorch3d.config import PATH_TYPE


class MetashapeCamera:
    def __init__(
        self,
        image_filename: PATH_TYPE,
        cam_to_world_transform: np.ndarray,
        f: float,
        cx: float,
        cy: float,
        image_width: int,
        image_height: int,
    ):
        """Represents the information about one camera location/image as determined by Metashape

        Args:
            image_filename (PATH_TYPE): The image used for reconstruction
            transform (np.ndarray): A 4x4 transform representing the camera-to-world transform
            f (float): Focal length in pixels
            cx (float): Principle point x (pixels)
            cy (float): Principle point y (pixels)
            image_width (int): Input image width pixels
            image_height (int): Input image height pixels
        """
        self.image_filename = image_filename
        self.cam_to_world_transform = cam_to_world_transform
        self.world_to_cam_transform = np.linalg.inv(cam_to_world_transform)
        self.f = f
        self.cx = cx
        self.cy = cy
        self.image_width = image_width
        self.image_height = image_height

        self.image = None
        self.cache_image = True

    def load_image(self, image_scale: float = 1.0) -> np.ndarray:
        # Check if the image is cached
        if self.image is not None:
            image = imread(self.image_filename)
            if image.dtype == np.uint8:
                image = image / 255.0

            # Avoid unneccesary read if we have memory
            if self.cache_image:
                self.image = image
        else:
            image = self.image

        # Resizing is never cached, consider revisiting
        if image_scale != 1.0:
            image = resize(
                image,
                (int(image.shape[0] * image_scale), int(image.shape[1] * image_scale)),
            )

        return image

    def check_projected_in_image(
        self, homogenous_image_coords: np.ndarray, image_size: Tuple[int, int]
    ):
        """Check if projected points are within the bound of the image and in front of camera

        Args:
            homogenous_image_coords (np.ndarray): The points after the application of K[R|t]. (3, n_points)
            image_size (Tuple[int, int]): The size of the image (width, height) in pixels

        Returns:
            np.ndarray: valid_points_bool, boolean array corresponding to which points were valid (n_points)
            np.ndarray: valid_image_space_points, float array of image-space coordinates for only valid points, (n_valid_points, 2)
        """
        img_width, image_height = image_size

        # Divide by the z coord to project onto the image plane
        image_space_points = homogenous_image_coords[:2] / homogenous_image_coords[2:3]
        # Transpose for convenience, (n_points, 3)
        image_space_points = image_space_points.T

        # We only want to consider points in front of the camera. Simple projection cannot tell
        # if a point is on the same ray behind the camera
        in_front_of_cam = homogenous_image_coords[2] > 0

        # Check that the point is projected within the image and is in front of the camera
        # Pytorch doesn't have a logical_and.reduce operator, so this is the equivilent using boolean multiplication
        valid_points_bool = (
            (image_space_points[:, 0] > 0)
            * (image_space_points[:, 1] > 0)
            * (image_space_points[:, 0] < img_width)
            * (image_space_points[:, 1] < image_height)
            * in_front_of_cam
        )

        # Extract the points that are valid
        valid_image_space_points = image_space_points[valid_points_bool, :].to(
            torch.int
        )
        # Return the boolean array
        valid_points_bool = valid_points_bool.cpu().numpy()
        valid_image_space_points = valid_image_space_points.cpu().numpy
        return valid_points_bool, valid_image_space_points

    def extract_colors(
        self, valid_bool: np.ndarray, valid_locs: np.ndarray, img: np.ndarray
    ):
        """_summary_

        Args:
            valid_bool (np.ndarray): (n_points,) boolean array cooresponding to valid points
            valid_locs (np.ndarray): (n_valid, 2) float array of image-space locations (x,y)
            img (np.ndarray): (h, w, n_channels) image to query from

        Returns:
            np.ma.array: (n_points, n_channels) One color per valid vertex. Points that were invalid are masked out
        """
        # Set up the data arrays
        colors_per_vertex = np.zeros((valid_bool.shape[0], img.shape[2]))
        mask = np.ones((valid_bool.shape[0], img.shape[2])).astype(bool)

        # Set the entries which are valid to false, meaning a valid entry in the masked array
        # TODO see if I can use valid_bool directly instead
        valid_inds = np.where(valid_bool)[0]
        mask[valid_inds, :] = False

        # Extract coordinates
        i_locs = valid_locs[:, 1]
        j_locs = valid_locs[:, 0]
        # Index based on the coordinates
        valid_color_samples = img[i_locs, j_locs, :]
        # Insert the valid samples into the array at the valid locations
        colors_per_vertex[valid_inds, :] = valid_color_samples
        # Convert to a masked array
        masked_color_per_vertex = ma.array(colors_per_vertex, mask=mask)
        return masked_color_per_vertex

    def project_mesh_verts(self, mesh_verts: np.ndarray, img: np.ndarray, device: str):
        """Get a color per vertex using only projective geometry, without considering occlusion or distortion

        Returns:
            np.ma.array: (n_points, n_channels) One color per valid vertex. Points that were invalid are masked out
        """
        # [R|t] matrix
        transform_3x4_world_to_cam = torch.Tensor(
            self.world_to_cam_transform[:3, :]
        ).to(device)
        K = torch.Tensor(
            [
                [self.f, 0, self.image_width / 2.0 + self.cx],
                [0, self.f, self.image_width + self.cy],
                [0, 0, 1],
            ],
            device=device,
        )
        # K[R|t], (3,4). Premultiplying these two matrices avoids doing two steps of projections with all points
        camera_matrix = K @ transform_3x4_world_to_cam

        # Add the extra dimension of ones for matrix multiplication
        homogenous_mesh_verts = torch.concatenate(
            (
                torch.Tensor(mesh_verts).to(device),
                torch.ones((mesh_verts.shape[0], 1)).to(device),
            ),
            axis=1,
        ).T

        # TODO review terminology
        homogenous_camera_points = camera_matrix @ homogenous_mesh_verts
        # Determine what points project onto the image and at what locations
        valid_bool, valid_locs = self.check_projected_in_image(
            projected_verts=homogenous_camera_points,
            image_size=(self.image_width, self.image_height),
        )
        # Extract corresponding colors from the image
        colors_per_vertex = self.extract_colors(valid_bool, valid_locs, img)

        return colors_per_vertex

    def get_pytorch3d_camera(self, device: str):
        """Return a pytorch3d camera based on the parameters from metashape

        Args:
            device (str): What device (cuda/cpu) to put the object on

        Returns:
            pytorch3d.renderer.PerspectiveCameras:
        """
        rotation_about_z = np.array(
            [[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        )
        # Rotate about the Z axis because the NDC coordinates are defined X: left, Y: up and we use X: right, Y: down
        # See https://pytorch3d.org/docs/cameras
        transform_4x4_world_to_cam = rotation_about_z @ self.world_to_cam_transform

        R = torch.Tensor(np.expand_dims(transform_4x4_world_to_cam[:3, :3].T, axis=0))
        T = torch.Tensor(np.expand_dims(transform_4x4_world_to_cam[:3, 3], axis=0))

        # The image size is (height, width) which completely disreguards any other conventions they use...
        image_size = ((self.image_height, self.image_width),)
        # These parameters are in screen (pixel) coordinates.
        # TODO see if a normalized version is more robust for any reason
        fcl_screen = (self.f,)
        prc_points_screen = (
            (self.image_width / 2 + self.cx, self.image_height / 2 + self.cy),
        )

        # Create camera
        # TODO use the pytorch3d FishEyeCamera model that uses distortion
        # https://pytorch3d.readthedocs.io/en/latest/modules/renderer/fisheyecameras.html?highlight=distortion
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
        """
        Visualize the camera as a frustum, at the appropriate translation and
        rotation and with the given focal length and aspect ratio.


        Args:
            plotter (pv.Plotter): The plotter to add the visualization to
            frustum_scale (float, optional): The length of the frustum in world units. Defaults to 0.5.
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

        projected_vertices = self.cam_to_world_transform @ vertices
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
        """
        Create a camera set from a metashape .xml camera file and the path to the image folder


        Args:
            camera_file (PATH_TYPE): Path to the .xml camera export from Metashape
            image_folder (PATH_TYPE): Path to the folder of images used by Metashape
        """
        self.parse_metashape_cam_file(camera_file=camera_file)

        self.image_filenames = [
            str(list(Path(image_folder).glob(filename + "*"))[0])
            for filename in self.image_filenames
        ]  # Assume there's only one file with that extension
        self.cameras = []

        for image_filename, cam_to_world_transform in zip(
            self.image_filenames, self.cam_to_world_transforms
        ):
            new_camera = MetashapeCamera(
                image_filename,
                cam_to_world_transform,
                self.f,
                self.cx,
                self.cy,
                self.image_width,
                self.image_height,
            )
            self.cameras.append(new_camera)

    def get_camera_by_index(self, index: int) -> MetashapeCamera:
        if index >= len(self.cameras):
            raise ValueError("Requested camera ind larger than list")
        return self.cameras[index]

    def vis(self, plotter: pv.Plotter, add_orientation_cube: bool = False):
        """Visualize all the cameras

        Args:
            plotter (pv.Plotter): Plotter to add the cameras to
            add_orientation_cube (bool, optional): Add a cube to visualize the coordinate system. Defaults to False.
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
            plotter.show_axes()

    def make_4x4_transform(
        self, rotation_str: str, translation_str: str, scale_str: str = "1"
    ):
        """Convenience function to make a 4x4 matrix from the string format used by Metashape

        Args:
            rotation_str (str): Row major with 9 entries
            translation_str (str): 3 entries
            scale_str (str, optional): single value. Defaults to "1".

        Returns:
            np.ndarray: (4, 4) A homogenous transform mapping from cam to world
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
        """Parse the information about the camera intrinsics and extrinsics

        Args:
            camera_file (str): Path to metashape .xml export

        Raises:
            ValueError: If camera calibration does not contain the f, cx, and cy params
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

        self.image_filenames = []
        self.cam_to_world_transforms = []
        for camera in cameras:
            if len(camera) < 5:
                # skipping unaligned camera
                continue
            self.image_filenames.append(camera.get("label"))
            self.cam_to_world_transforms.append(
                np.fromstring(camera[0].text, sep=" ").reshape(4, 4)
            )
