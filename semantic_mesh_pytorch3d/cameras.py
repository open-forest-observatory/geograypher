import xml.etree.ElementTree as ET
import numpy as np
import pyvista as pv

class MetashapeCamera():
    def __init__(self, filename, transform, f, cx, cy, scale, image_width, image_height):
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

class MetashapeCameraSet():
    def __init__(self, camera_file):
        self.filenames, self.transforms, self.f, self.cx, self.cy, self.scale, self.image_width, self.image_height = self.parse_cam_file(camera_file)

        self.cameras = []

        for filename, transform in zip(self.filenames, self.transforms):
            new_camera = MetashapeCamera(filename, transform, self.f, self.cx, self.cy, self.scale, self.image_width, self.image_height)
            self.cameras.append(new_camera)

    def rescale(self, scale):
        for camera in self.cameras:
            camera.rescale(scale)

    def vis(self, plotter: pv.Plotter):
        for camera in self.cameras:
            camera.vis(plotter)


    def parse_cam_file(self,camera_file):
        # Load the xml file
        tree = ET.parse(camera_file)
        root = tree.getroot()
        # Iterate through the camera poses

        filenames = []
        transforms = []

        scale = float(root[0][1][0][0][2].text)
        for camera in root[0][2]:
            # Print the filename
            # Get the transform as a (16,) vector
            transform = camera[0].text
            if transform is None:
                continue
            transform = np.fromstring(transform, sep=" ")
            transform = transform.reshape((4, 4))
            # Fix the fact that the transforms are reported in a local scale
            transform[:3, 3] = transform[:3, 3] * scale
            filenames.append(camera.attrib["label"])
            transforms.append(transform)
        f = float(root[0][0][0][4][1].text)
        cx = float(root[0][0][0][4][2].text)
        cy = float(root[0][0][0][4][3].text)

        width = float(root[0][0][0][4][0].get("width"))
        height = float(root[0][0][0][4][0].get("height"))
        return filenames, transforms, f, cx, cy, scale, width, height



MetashapeCameraSet("/ofo-share/repos-david/Safeforest_CMU_data_dvc/data/site_Gascola/04_27_23/collect_05/processed_02/metashape/left_camera_automated/exports/example-run-001_20230517T1827_camera.xml")