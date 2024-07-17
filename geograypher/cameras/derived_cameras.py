import typing
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import pyproj
from pytorch3d.renderer import PerspectiveCameras
from scipy.spatial.transform import Rotation
import torch

from geograypher.cameras import PhotogrammetryCamera, PhotogrammetryCameraSet
from geograypher.constants import (
    EARTH_CENTERED_EARTH_FIXED_EPSG_CODE,
    LAT_LON_EPSG_CODE,
    PATH_TYPE,
    EXAMPLE_INTRINSICS,
)
from geograypher.utils.parsing import parse_sensors, parse_transform_metashape


def update_lists(
    camera,
    image_folder,
    cam_to_world_transforms,
    image_filenames,
    sensor_IDs,
):
    transform = camera.find("transform")
    if transform is None:
        # skipping unaligned camera
        return
    # If valid, parse into numpy array
    cam_to_world_transforms.append(np.fromstring(transform.text, sep=" ").reshape(4, 4))

    # The label should contain the image path
    # TODO see if we want to do any fixup here, or punt to automate-metashape
    image_filenames.append(Path(image_folder, camera.get("label")))
    # This says which sensor model it came from
    sensor_IDs.append(int(camera.get("sensor_id")))
    # Try to get the lat lon information


class MetashapeCameraSet(PhotogrammetryCameraSet):
    def __init__(
        self,
        camera_file: PATH_TYPE,
        image_folder: PATH_TYPE,
        validate_images: bool = False,
        default_sensor_params: dict = {},
    ):
        """Parse the information about the camera intrinsics and extrinsics

        Args:
            camera_file (PATH_TYPE): Path to metashape .xml export
            image_folder: (PATH_TYPE): Path to image folder root

        Raises:
            ValueError: If camera calibration does not contain the f, cx, and cy params
        """
        # Load the xml file
        # Taken from here https://rowelldionicio.com/parsing-xml-with-python-minidom/
        tree = ET.parse(camera_file)
        root = tree.getroot()
        # first level
        chunk = root.find("chunk")
        # second level
        sensors = chunk.find("sensors")
        # Parse the sensors representation
        sensors_dict = parse_sensors(sensors, default_sensor_dict=default_sensor_params)

        # Set up the lists to populate
        image_filenames = []
        cam_to_world_transforms = []
        sensor_IDs = []

        cameras = chunk[2]
        # Iterate over metashape cameras and fill out required information
        for cam_or_group in cameras:
            if cam_or_group.tag == "group":
                for cam in cam_or_group:
                    update_lists(
                        cam,
                        image_folder,
                        cam_to_world_transforms,
                        image_filenames,
                        sensor_IDs,
                    )
            else:
                # 4x4 transform
                update_lists(
                    cam_or_group,
                    image_folder,
                    cam_to_world_transforms,
                    image_filenames,
                    sensor_IDs,
                )

        # Compute the lat lon using the transforms, because the reference values recorded in the file
        # reflect the EXIF values, not the optimized ones

        # Get the transform from the chunk to the earth-centered, earth-fixed (ECEF) frame
        chunk_to_epsg4327 = parse_transform_metashape(camera_file=camera_file)

        if chunk_to_epsg4327 is not None:
            # Compute the location of each camera in ECEF
            cam_locs_in_epsg4327 = []
            for cam_to_world_transform in cam_to_world_transforms:
                cam_loc_in_chunk = cam_to_world_transform[:, 3:]
                cam_locs_in_epsg4327.append(chunk_to_epsg4327 @ cam_loc_in_chunk)
            cam_locs_in_epsg4327 = np.concatenate(cam_locs_in_epsg4327, axis=1)[:3].T
            # Transform these points into lat-lon-alt
            transformer = pyproj.Transformer.from_crs(
                EARTH_CENTERED_EARTH_FIXED_EPSG_CODE, LAT_LON_EPSG_CODE
            )
            lat, lon, _ = transformer.transform(
                xx=cam_locs_in_epsg4327[:, 0],
                yy=cam_locs_in_epsg4327[:, 1],
                zz=cam_locs_in_epsg4327[:, 2],
            )
            lon_lats = list(zip(lon, lat))
        else:
            # TODO consider trying to parse from the xml
            lon_lats = None

        # Actually construct the camera objects using the base class
        super().__init__(
            cam_to_world_transforms=cam_to_world_transforms,
            intrinsic_params_per_sensor_type=sensors_dict,
            image_filenames=image_filenames,
            lon_lats=lon_lats,
            image_folder=image_folder,
            sensor_IDs=sensor_IDs,
            validate_images=validate_images,
        )


class COLMAPCameraSet(PhotogrammetryCameraSet):

    def __init__(
        self,
        cameras_file: PATH_TYPE,
        images_file: PATH_TYPE,
        image_folder: typing.Union[None, PATH_TYPE] = None,
        validate_images: bool = False,
    ):
        """
        Create a camera set from the files exported by the open-source structure-from-motion
        software COLMAP as defined here: https://colmap.github.io/format.html

        Args:
            cameras_file (PATH_TYPE):
                Path to the file containing the camera models definitions
            images_file (PATH_TYPE):
                Path to the per-image information, including the pose and which camera model is used
            image_folder (typing.Union[None, PATH_TYPE], optional):
                Path to the folder of images used to generate the reconstruction. Defaults to None.
            validate_images (bool, optional):
                Ensure that the images described in images_file are present in image_folder.
                Defaults to False.

        Raises:
            NotImplementedError: If the camera is not a Simple radial model
        """
        # Parse the csv representation of the camera models
        cameras_data = pd.read_csv(
            cameras_file,
            sep=" ",
            skiprows=[0, 1, 2],
            header=None,
            names=(
                "CAMERA_ID",
                "MODEL",
                "WIDTH",
                "HEIGHT",
                "PARAMS_F",
                "PARAMS_CX",
                "PARAMS_CY",
                "PARAMS_RADIAL",
            ),
        )
        # Parse the csv of the per-image information
        # Note that every image has first the useful information on one row and then unneeded
        # keypoint information on the following row. Therefore, the keypoints are discarded.
        images_data = pd.read_csv(
            images_file,
            sep=" ",
            skiprows=lambda x: (x in (0, 1, 2, 3) or x % 2),
            header=None,
            names=(
                "IMAGE_ID",
                "QW",
                "QX",
                "QY",
                "QZ",
                "TX",
                "TY",
                "TZ",
                "CAMERA_ID",
                "NAME",
            ),
            usecols=list(range(10)),
        )

        # TODO support more camera models
        if np.any(cameras_data["MODEL"] != "SIMPLE_RADIAL"):
            raise NotImplementedError("Not a supported camera model")

        # Parse the camera parameters, creating a dict for each distinct camera model
        sensors_dict = {}
        for _, row in cameras_data.iterrows():
            # Note that the convention in this tool is for cx, cy to be defined from the center
            # not the corner so it must be shifted
            sensor_dict = {
                "image_width": row["WIDTH"],
                "image_height": row["HEIGHT"],
                "f": row["PARAMS_F"],
                "cx": row["PARAMS_CX"] - row["WIDTH"] / 2,
                "cy": row["PARAMS_CY"] - row["HEIGHT"] / 2,
                "distortion_params": {"r": row["PARAMS_RADIAL"]},
            }
            sensors_dict[row["CAMERA_ID"]] = sensor_dict

        # Parse the per-image information
        cam_to_world_transforms = []
        sensor_IDs = []
        image_filenames = []

        for _, row in images_data.iterrows():
            # Convert from the quaternion representation to the matrix one. Note that the W element
            # is the first one in the COLMAP convention but the last one in scipy.
            rot_mat = Rotation.from_quat(
                (row["QX"], row["QY"], row["QZ"], row["QW"])
            ).as_matrix()
            # Get the camera translation
            translation_vec = np.array([row["TX"], row["TY"], row["TZ"]])

            # Create a 4x4 homogenous matrix representing the world_to_cam transform
            world_to_cam = np.eye(4)
            # Populate the sub-elements
            world_to_cam[:3, :3] = rot_mat
            world_to_cam[:3, 3] = translation_vec
            # We need the cam to world transform. Since we're using a 4x4 representation, we can
            # just invert the matrix
            cam_to_world = np.linalg.inv(world_to_cam)
            cam_to_world_transforms.append(cam_to_world)

            # Record which camera model is used and the image filename
            sensor_IDs.append(row["CAMERA_ID"])
            image_filenames.append(Path(image_folder, row["NAME"]))

        # Instantiate the camera set
        super().__init__(
            cam_to_world_transforms=cam_to_world_transforms,
            intrinsic_params_per_sensor_type=sensors_dict,
            image_filenames=image_filenames,
            sensor_IDs=sensor_IDs,
            image_folder=image_folder,
            validate_images=validate_images,
        )

class PyTorch3DRenderingCamera(PhotogrammetryCamera):
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

class PyTorch3DRenderingCameraSet(PhotogrammetryCameraSet):
    def __init__(
        self,
        cameras: Union[None, PhotogrammetryCamera, List[PhotogrammetryCamera]] = None,
        cam_to_world_transforms: Union[None, List[np.ndarray]] = None,
        intrinsic_params_per_sensor_type: Dict[int, Dict[str, float]] = {
            0: EXAMPLE_INTRINSICS
        },
        image_filenames: Union[List[PATH_TYPE], None] = None,
        lon_lats: Union[None, List[Union[None, Tuple[float, float]]]] = None,
        image_folder: PATH_TYPE = None,
        sensor_IDs: List[int] = None,
        validate_images: bool = False,
        class_type = PyTorch3DRenderingCamera,
    ):
        super().__init__(cameras, cam_to_world_transforms, intrinsic_params_per_sensor_type,image_filenames, lon_lats, image_folder, sensor_IDs, validate_images, class_type)
        
    def get_pytorch3d_camera(self, device: str, dervied_cameras):
        """
        Return a pytorch3d cameras object based on the parameters from metashape.
        This has the information from each of the camears in the set to enabled batched rendering.


        Args:
            device (str): What device (cuda/cpu) to put the object on

        Returns:
            pytorch3d.renderer.PerspectiveCameras:
        """
        # Get the pytorch3d cameras for each of the cameras in the set
        # p3d_cameras = [camera.get_pytorch3d_camera(device) for camera in self.cameras]
        p3d_cameras = [camera.get_pytorch3d_camera(device) for camera in dervied_cameras]
        # Get the image sizes
        image_sizes = [camera.image_size.cpu().numpy() for camera in p3d_cameras]
        # Check that all the image sizes are the same because this is required for proper batched rendering
        if np.any([image_size != image_sizes[0] for image_size in image_sizes]):
            raise ValueError("Not all cameras have the same image size")
        # Create the new pytorch3d cameras object with the information from each camera
        cameras = PerspectiveCameras(
            R=torch.cat([camera.R for camera in p3d_cameras], 0),
            T=torch.cat([camera.T for camera in p3d_cameras], 0),
            focal_length=torch.cat([camera.focal_length for camera in p3d_cameras], 0),
            principal_point=torch.cat(
                [camera.get_principal_point() for camera in p3d_cameras], 0
            ),
            device=device,
            in_ndc=False,  # screen coords
            image_size=image_sizes[0],
        )
        return cameras