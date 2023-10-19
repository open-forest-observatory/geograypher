import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from multiview_prediction_toolkit.cameras import PhotogrammetryCameraSet
from multiview_prediction_toolkit.config import PATH_TYPE


class MetashapeCameraSet(PhotogrammetryCameraSet):
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

        # sensors info
        # TODO in the future we should support multiple sensors
        # This would required parsing multiple sensor configs and figuring out
        # which image each corresponds to
        sensor = sensors[0]
        self.image_width = int(sensor[0].get("width"))
        self.image_height = int(sensor[0].get("height"))

        calibration = sensor.find("calibration")
        if calibration is None:
            raise ValueError("No calibration provided")

        self.f = float(calibration.find("f").text)
        self.cx = float(calibration.find("cx").text)
        self.cy = float(calibration.find("cy").text)
        if None in (self.f, self.cx, self.cy):
            ValueError("Incomplete calibration provided")

        # Get potentially-empty dict of distortion parameters
        self.distortion_dict = {
            calibration[i].tag: float(calibration[i].text)
            for i in range(3, len(calibration))
        }

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

        self.image_filenames = [
            str(list(Path(image_folder).glob(filename + "*"))[0])
            for filename in self.image_filenames
        ]  # Assume there's only one file with that extension

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
