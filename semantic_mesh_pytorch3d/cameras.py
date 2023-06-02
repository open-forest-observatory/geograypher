import xml.etree.ElementTree as ET
import numpy as np
import pyvista as pv


class MetashapeCameraSet():
    def __init__(self, camera_file):
        breakpoint()

    def load_cameras(camera_file):
        # Load the xml file
        tree = ET.parse(camera_file)
        root = tree.getroot()
        # Iterate through the camera poses

        labels = []
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
            labels.append(camera.attrib["label"])
            transforms.append(transform)
        f = float(root[0][0][0][4][1].text)
        cx = float(root[0][0][0][4][2].text)
        cy = float(root[0][0][0][4][3].text)

        width = float(root[0][0][0][4][0].get("width"))
        height = float(root[0][0][0][4][0].get("height"))

        return labels, np.array(transforms), f, cx, cy, scale, width, height

class MetashapeCamera():
    def __init__(self):
        breakpoint()

MetashapeCameraSet("")