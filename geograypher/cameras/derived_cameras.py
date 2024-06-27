import typing
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
import pyproj
from scipy.spatial.transform import Rotation

from geograypher.cameras import PhotogrammetryCameraSet
from geograypher.constants import (
    EARTH_CENTERED_EARTH_FIXED_EPSG_CODE,
    LAT_LON_EPSG_CODE,
    PATH_TYPE,
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
