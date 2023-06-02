import xml.etree.ElementTree as ET
import numpy as np
import pyvista as pv
import xml.dom.minidom


class MetashapeCamera:
    def __init__(
        self, filename, transform, f, cx, cy, scale, image_width, image_height
    ):
        self.filename = filename
        self.transform = transform
        self.f = f
        self.cx = cx
        self.cy = cy
        self.scale = scale
        self.image_width = image_width
        self.image_height = image_height

    def rescale(self, scale: float):
        """
        Rescale transform in place
        """
        self.transform[:3, 3] = self.transform[:3, 3] * scale

    def vis(self, plotter: pv.Plotter):
        camera_loc = pv.PolyData(self.transform[:3, 3:].T)
        plotter.add_mesh(camera_loc)


class MetashapeCameraSet:
    def __init__(self, camera_file):
        (
            self.filenames,
            self.transforms,
            self.f,
            self.cx,
            self.cy,
            self.scale,
            self.image_width,
            self.image_height,
        ) = self.parse_cam_file(camera_file)

        self.cameras = []

        for filename, transform in zip(self.filenames, self.transforms):
            new_camera = MetashapeCamera(
                filename,
                transform,
                self.f,
                self.cx,
                self.cy,
                self.scale,
                self.image_width,
                self.image_height,
            )
            self.cameras.append(new_camera)

    def rescale(self, scale):
        for camera in self.cameras:
            camera.rescale(scale)

    def vis(self, plotter: pv.Plotter):
        for camera in self.cameras:
            camera.vis(plotter)

    def make_4x4_transform(self, rotation_str, translation_str, scale_str):
        rotation_np = np.fromstring(rotation_str, sep=" ")
        rotation_np = np.reshape(rotation_np, (3, 3))
        translation_np = np.fromstring(translation_str, sep=" ")
        scale = float(scale_str)
        transform = np.zeros((4, 4))
        transform[:3, :3] = rotation_np
        transform[:3, 3] = translation_np
        transform[3, 3] = scale
        return transform

    def parse_cam_file(self, camera_file):
        # Load the xml file
        # Taken from here https://rowelldionicio.com/parsing-xml-with-python-minidom/
        tree = ET.parse(camera_file)
        root = tree.getroot()
        # first level
        chunk = root[0]
        # second level
        sensors = chunk[0]
        cameras = chunk[2]
        transform = chunk[3]
        region = chunk[6]

        # transform info
        global_rotation = transform[0].text
        global_translation = transform[1].text
        global_scale = transform[2].text
        global_transform = self.make_4x4_transform(
            global_rotation, global_translation, global_scale
        )

        # sensors info
        sensor = sensors[0]
        calibration = sensor[4]
        f = float(calibration[1].text)
        cx = float(calibration[2].text)
        cy = float(calibration[3].text)

        width = float(calibration[0].get("width"))
        height = float(calibration[0].get("height"))

        # region info
        region_translation = region[0].text
        # size = np.fromstring(region[1].text, sep=" ")
        region_rotation = region[2].text
        region_transform = self.make_4x4_transform(
            region_rotation, region_translation, global_scale
        )

        # Iterate through the camera poses

        filenames = []
        transforms = []

        for camera in cameras:
            # Print the filename
            # Get the transform as a (16,) vector
            transform = camera[0].text
            if transform is None:
                continue
            transform = np.fromstring(transform, sep=" ")
            transform = transform.reshape((4, 4))
            # Fix the fact that the transforms are reported in a local scale
            transform[:3, 3] = transform[:3, 3]
            transform = region_transform @ transform
            filenames.append(camera.attrib["label"])
            transforms.append(transform)
        return filenames, transforms, f, cx, cy, global_scale, width, height
