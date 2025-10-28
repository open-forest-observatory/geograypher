import typing
import xml.etree.ElementTree as ET
from copy import copy
from pathlib import Path

import numpy as np
import pyproj


def parse_metashape_mesh_metadata(
    mesh_metadata_file: typing.Union[str, Path],
) -> typing.Tuple[typing.Union[pyproj.CRS, None], typing.Union[np.ndarray, None]]:
    """
    Parse the metadata file which is produced by Metashape when exporting a mesh to determine the
    coordinate reference frame and origin shift to use when interpreting the mesh.


    Args:
        mesh_metadata_file (typing.Union[str, Path]): Path to the metadata XML file.

    Returns:
        Union[pyproj.CRS, None]:
            The CRS to interpret the vertices (after the shift). If not present, None is returned.
        Union[np.ndarray, None]:
            The shift to be added to the mesh vertices. If not present, None is returned.
    """
    tree = ET.parse(mesh_metadata_file)
    root = tree.getroot()

    # Parse CRS and shift
    CRS = root.find("SRS")
    shift = root.find("SRSOrigin")

    # If CRS is present, convert it to a pyproj type
    if CRS is not None:
        CRS = pyproj.CRS(CRS.text)

    # If the shift is present, convert to a numpy array
    if shift is not None:
        shift = np.array(shift.text.split(","), dtype=float)
    return CRS, shift


def make_4x4_transform(rotation_str: str, translation_str: str, scale_str: str = "1"):
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

    if not np.isclose(np.linalg.det(rotation_np), 1.0, atol=1e-8, rtol=0):
        raise ValueError(
            f"Inproper rotation matrix with determinant {np.linalg.det(rotation_np)}"
        )

    translation_np = np.fromstring(translation_str, sep=" ")
    scale = float(scale_str)
    transform = np.eye(4)
    transform[:3, :3] = rotation_np * scale
    transform[:3, 3] = translation_np
    return transform


def parse_transform_metashape(camera_file):
    tree = ET.parse(camera_file)
    root = tree.getroot()
    # first level
    components = root.find("chunk").find("components")

    assert len(components) == 1
    transform = components.find("component").find("transform")
    if transform is None:
        return None

    rotation = transform.find("rotation").text
    translation = transform.find("translation").text
    scale = transform.find("scale").text

    local_to_epgs_4978_transform = make_4x4_transform(rotation, translation, scale)

    return local_to_epgs_4978_transform


def parse_sensors(sensors, default_sensor_dict=None):
    sensors_dict = {}

    for sensor in sensors:
        sensor_dict = {}

        sensor_dict["image_width"] = int(sensor[0].get("width"))
        sensor_dict["image_height"] = int(sensor[0].get("height"))
        calibration = sensor.find("calibration[@class='adjusted']")

        if calibration is None:
            if default_sensor_dict is not None:
                for k, v in default_sensor_dict.items():
                    sensor_dict[k] = v
            else:
                sensor_dict = None
        else:
            sensor_dict["f"] = float(calibration.find("f").text)

            # Extract the objects for the principal points
            cx = calibration.find("cx")
            cy = calibration.find("cy")

            try:
                # If they are set, use the value. Otherwise, use the default.
                # TODO handle the case where no default is provided explicitly
                sensor_dict["cx"] = (
                    float(cx.text) if cx is not None else default_sensor_dict["cx"]
                )
                sensor_dict["cy"] = (
                    float(cy.text) if cy is not None else default_sensor_dict["cy"]
                )
                # Get potentially-empty dict of distortion parameters
                sensor_dict["distortion_params"] = {
                    calibration[i].tag: float(calibration[i].text)
                    for i in range(len(calibration))
                    if calibration[i].tag not in ["resolution", "f", "cx", "cy"]
                }
            except KeyError:
                sensor_dict = None

        sensor_ID = int(sensor.get("id"))
        sensors_dict[sensor_ID] = sensor_dict
    return sensors_dict
