import xml.etree.ElementTree as ET
from pathlib import Path
from glob import glob

import numpy as np

from multiview_prediction_toolkit.cameras import PhotogrammetryCameraSet
from multiview_prediction_toolkit.config import PATH_TYPE
from multiview_prediction_toolkit.utils.parsing import parse_transform_metashape


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

        self.local_to_epgs_4978_transform = parse_transform_metashape(camera_file)

        cameras = chunk[2]

        self.cam_to_world_transforms = []

        camera_labels = [camera.get("label") for camera in cameras]

        self.image_filenames = self.get_absolute_filenames(image_folder, camera_labels)
        self.lon_lats = []

        for camera in cameras:
            transform = camera.find("transform")
            if transform is None:
                # skipping unaligned camera
                continue
            self.cam_to_world_transforms.append(
                np.fromstring(transform.text, sep=" ").reshape(4, 4)
            )
            reference = camera.find("reference")
            lon_lat = (float(reference.get("x")), float(reference.get("y")))
            self.lon_lats.append(lon_lat)
        # <reference x="-120.087143111111" y="38.967084472222197" z="2084.4450000000002" yaw="5.8999999999999995" pitch="0.099999999999993788" roll="-0" sxyz="62" enabled="false"/>

    def get_absolute_filenames(self, image_folder, camera_labels, image_extension=""):
        absolute_filenames = []

        for camera_label in camera_labels:
            if camera_label.split("/")[0] == "ofo-share":
                absolute_filename = Path("/", camera_label)
            else:
                search_str = str(Path(image_folder, "**", camera_label))
                matching_files = list(glob(search_str))
                if len(matching_files) != 1 and False:
                    raise ValueError(
                        f"Bad match for {search_str} resulted in {len(matching_files)} files"
                    )
                absolute_filename = Path(matching_files[0])
            absolute_filenames.append(absolute_filename)

        return absolute_filenames
