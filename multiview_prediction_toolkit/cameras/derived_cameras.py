import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from torch import Value

from multiview_prediction_toolkit.cameras import (
    PhotogrammetryCameraSet,
    PhotogrammetryCamera,
)
from multiview_prediction_toolkit.config import PATH_TYPE

from multiview_prediction_toolkit.utils.parsing import (
    parse_sensors,
    parse_transform_metashape,
)


class MetashapeCameraSet(PhotogrammetryCameraSet):
    def __init__(self, camera_file: PATH_TYPE, image_folder: PATH_TYPE, **kwargs):
        """
        Create a camera set from a metashape .xml camera file and the path to the image folder


        Args:
            camera_file (PATH_TYPE): Path to the .xml camera export from Metashape
            image_folder (PATH_TYPE): Path to the folder of images used by Metashape
        """
        self.parse_input(camera_file=camera_file, image_folder=image_folder, **kwargs)

        self.cameras = []

        for image_filename, cam_to_world_transform, sensor_id in zip(
            self.image_filenames, self.cam_to_world_transforms, self.sensor_IDs
        ):
            sensor_dict = self.sensors_dict[sensor_id]
            new_camera = PhotogrammetryCamera(
                image_filename, cam_to_world_transform, **sensor_dict
            )
            self.cameras.append(new_camera)

    def parse_input(self, camera_file: PATH_TYPE, image_folder: PATH_TYPE):
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

        # Get the transform relating the arbitrary local coordinate system
        # to the earth-centered earth-fixed EPGS:4978 system that is used as a reference by metashape

        self.local_to_epgs_4978_transform = parse_transform_metashape(camera_file)

        cameras = chunk[2]

        self.image_filenames = []
        self.cam_to_world_transforms = []
        self.sensor_IDs = []

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
