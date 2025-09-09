import typing
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
import pyproj
from scipy.spatial.transform import Rotation

from geograypher.cameras import PhotogrammetryCamera, PhotogrammetryCameraSet
from geograypher.constants import EARTH_CENTERED_EARTH_FIXED_CRS, LAT_LON_CRS, PATH_TYPE
from geograypher.utils.parsing import parse_sensors, parse_transform_metashape


def update_lists(
    camera,
    image_folder,
    cam_to_world_transforms,
    image_filenames,
    sensor_IDs,
    original_image_folder=None,
):
    transform = camera.find("transform")
    if transform is None:
        # skipping unaligned camera
        return
    # If valid, parse into numpy array
    cam_to_world_transforms.append(np.fromstring(transform.text, sep=" ").reshape(4, 4))

    # The label should contain the image path
    # Get the image filename stored in the label field
    image_filename = Path(camera.get("label"))
    if original_image_folder is not None:
        # The original paths are often absolute, corresponding to where the images used for metashape
        # were. The new images may be different, so we can make the path relative to this folder.
        image_filename = image_filename.relative_to(original_image_folder)
    # Prepend the current path of the images on disk
    image_filenames.append(Path(image_folder, image_filename))
    # This says which sensor model it came from
    sensor_IDs.append(int(camera.get("sensor_id")))
    # Try to get the lat lon information


class MetashapeCameraSet(PhotogrammetryCameraSet):
    def __init__(
        self,
        camera_file: PATH_TYPE,
        image_folder: PATH_TYPE,
        original_image_folder: typing.Optional[PATH_TYPE] = None,
        validate_images: bool = False,
        default_sensor_params: dict = {"cx": 0.0, "cy": 0.0},
    ):
        """Parse the information about the camera intrinsics and extrinsics

        Args:
            camera_file (PATH_TYPE):
                Path to metashape .xml export
            image_folder (PATH_TYPE):
                Path to image folder root
            original_image_folder (PATH_TYPE, optional):
                Path to where the original images for photogrammetry were, which was not included
                in the stored image zip files. This is removed from the absolute path recorded in
                the camera file. Defaults to None.
            validate_images (bool, optional): Should the existance of the images be checked.
                Any image_filenames found in the camera_file that do not exist on disk will be
                dropped, leaving a CameraSet only containing existing images. Defaults to False.
            default_sensor_params (dict, optional):
                Default parameters for the intrinsic parameters if not present. Defaults to zeros
                "cx" and "cy".

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

        cameras = chunk.find("cameras")
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
                        original_image_folder=original_image_folder,
                    )
            else:
                update_lists(
                    cam_or_group,
                    image_folder,
                    cam_to_world_transforms,
                    image_filenames,
                    sensor_IDs,
                    original_image_folder=original_image_folder,
                )

        # Compute the lat lon using the transforms, because the reference values recorded in the file
        # reflect the EXIF values, not the optimized ones

        # Get the transform from the chunk to the earth-centered, earth-fixed (ECEF) frame
        chunk_to_epsg4978 = parse_transform_metashape(camera_file=camera_file)

        if chunk_to_epsg4978 is not None:
            # Compute the location of each camera in ECEF
            cam_locs_in_epsg4978 = []
            for cam_to_world_transform in cam_to_world_transforms:
                cam_loc_in_chunk = cam_to_world_transform[:, 3:]
                cam_locs_in_epsg4978.append(chunk_to_epsg4978 @ cam_loc_in_chunk)
            cam_locs_in_epsg4978 = np.concatenate(cam_locs_in_epsg4978, axis=1)[:3].T
            # Transform these points into lat-lon-alt
            transformer = pyproj.Transformer.from_crs(
                EARTH_CENTERED_EARTH_FIXED_CRS, LAT_LON_CRS
            )
            lat, lon, _ = transformer.transform(
                xx=cam_locs_in_epsg4978[:, 0],
                yy=cam_locs_in_epsg4978[:, 1],
                zz=cam_locs_in_epsg4978[:, 2],
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
            local_to_epsg_4978_transform=chunk_to_epsg4978,
        )

    def ideal_to_warped(
        self, camera: PhotogrammetryCamera, xpix: np.ndarray, ypix: np.ndarray
    ) -> typing.Tuple[np.ndarray, np.ndarray]:
        """
        For consistency, this should fully match the docstring from
        cameras.PhotogrammetryCameraSet.ideal_to_warped()
        """

        # Convert from x and y pixels to the homogeneous camera frame
        # Note that for the math to work out, the cx and cy terms are neglected in this step and
        # only applied at the very end
        principal_x = camera.image_width / 2.0
        principal_y = camera.image_height / 2.0
        x = (xpix - principal_x) / camera.f
        y = (ypix - principal_y) / camera.f

        # Enforce that a strict subset of expected parameters are found
        params = sorted(camera.distortion_params.keys())
        if not set(params) <= set(["b1", "b2", "k1", "k2", "k3", "k4", "p1", "p2"]):
            raise ValueError(f"Unexpected distortion params found: {params}")
        b1 = camera.distortion_params.get("b1", 0)
        b2 = camera.distortion_params.get("b2", 0)
        k1 = camera.distortion_params["k1"]  # Enforce that the most basic is required
        k2 = camera.distortion_params.get("k2", 0)
        k3 = camera.distortion_params.get("k3", 0)
        k4 = camera.distortion_params.get("k4", 0)
        p1 = camera.distortion_params.get("p1", 0)
        p2 = camera.distortion_params.get("p2", 0)

        # See page 246 of the manual (labeled page 240) "Frame Cameras" section
        # for what these parameters mean
        # https://www.agisoft.com/pdf/metashape-pro_2_2_en.pdf
        r = np.sqrt(x**2 + y**2)
        #  Distorted rays
        xd = x * (1 + k1 * r**2 + k2 * r**4 + k3 * r**6 + k4 * r**8) + (
            p1 * (r**2 + 2 * x**2) + 2 * p2 * x * y
        )
        yd = y * (1 + k1 * r**2 + k2 * r**4 + k3 * r**6 + k4 * r**8) + (
            p2 * (r**2 + 2 * y**2) + 2 * p1 * x * y
        )
        # Pixels
        xpix_warp = (
            camera.image_width / 2.0 + camera.cx + xd * camera.f + xd * b1 + yd * b2
        )
        ypix_warp = camera.image_height / 2.0 + camera.cy + yd * camera.f
        return xpix_warp, ypix_warp


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
