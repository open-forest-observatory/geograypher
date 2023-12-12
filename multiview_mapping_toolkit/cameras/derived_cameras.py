import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from multiview_mapping_toolkit.cameras import PhotogrammetryCameraSet
from multiview_mapping_toolkit.config import PATH_TYPE
from multiview_mapping_toolkit.utils.parsing import (
    parse_sensors,
    parse_transform_metashape,
)


class MetashapeCameraSet(PhotogrammetryCameraSet):
    def parse_input(
        self, camera_file: PATH_TYPE, image_folder: PATH_TYPE, default_focal=None
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
        self.sensors_dict = parse_sensors(sensors)

        self.local_to_epgs_4978_transform = parse_transform_metashape(camera_file)

        cameras = chunk[2]

        self.image_filenames = []
        self.cam_to_world_transforms = []
        self.sensor_IDs = []
        self.lon_lats = []

        for camera in cameras:
            transform = camera.find("transform")
            if transform is None:
                # skipping unaligned camera
                continue
            self.image_filenames.append(Path(image_folder, camera.get("label")))
            self.cam_to_world_transforms.append(
                np.fromstring(transform.text, sep=" ").reshape(4, 4)
            )
            self.sensor_IDs.append(int(camera.get("sensor_id")))
            reference = camera.find("reference")
            lon_lat = (float(reference.get("x")), float(reference.get("y")))
            self.lon_lats.append(lon_lat)

        return (
            self.image_filenames,
            self.cam_to_world_transforms,
            self.sensor_IDs,
            self.lon_lats,
            self.sensors_dict,
        )
