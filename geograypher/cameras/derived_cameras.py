import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from geograypher.cameras import PhotogrammetryCameraSet
from geograypher.constants import PATH_TYPE
from geograypher.utils.parsing import (
    parse_sensors,
    parse_transform_metashape,
)


def update_lists(
    camera,
    image_folder,
    cam_to_world_transforms,
    image_filenames,
    sensor_IDs,
    lon_lats,
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
    reference = camera.find("reference")
    try:
        lon_lat = (float(reference.get("x")), float(reference.get("y")))
        # TODO see what error this would throw if those aren't set
    except:
        lon_lat = None

    lon_lats.append(lon_lat)


class MetashapeCameraSet(PhotogrammetryCameraSet):
    def __init__(
        self,
        camera_file: PATH_TYPE,
        image_folder: PATH_TYPE,
        validate_images: bool = False,
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
        sensors_dict = parse_sensors(sensors)

        # Set up the lists to populate
        image_filenames = []
        cam_to_world_transforms = []
        sensor_IDs = []
        lon_lats = []

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
                        lon_lats,
                    )
            else:
                # 4x4 transform
                update_lists(
                    cam_or_group,
                    image_folder,
                    cam_to_world_transforms,
                    image_filenames,
                    sensor_IDs,
                    lon_lats,
                )

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
