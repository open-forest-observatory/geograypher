import operator
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from geograypher.cameras.derived_cameras import MetashapeCameraSet

# Example portion of a real camera file
CAMERA_XML = """<?xml version="1.0" encoding="UTF-8"?>
<document version="2.0.0">
  <chunk label="Chunk 1" enabled="true">
    <sensors next_id="1">
      <sensor id="0" label="M3M (12.29mm)" type="frame">
        <resolution width="5280" height="3956"/>
        <property name="pixel_width" value="0.00335820"/>
        <property name="pixel_height" value="0.00335820"/>
        <property name="focal_length" value="12.28999999"/>
        <property name="layer_index" value="0"/>
        <calibration type="frame" class="adjusted">
          <resolution width="5280" height="3956"/>
          <f>3705.4728792737214</f>
          <cx>11.6738523909</cx>
          <cy>-27.7497969199</cy>
          <b1>0.5262024073</b1>
          <b2>-0.3058334293</b2>
          <k1>-0.0919367147</k1>
          <k2>-0.0762807468</k2>
          <k3>0.1162639394</k3>
          <k4>-0.0761413904</k4>
          <p1>-0.0003134847</p1>
          <p2>0.0001164035</p2>
        </calibration>
      </sensor>
    </sensors>
    <components next_id="1" active_id="0">
      <component id="0" label="Component 1">
        <transform>
          <rotation locked="true">0.8710699454 0.3070967021 -0.3833128821 -0.4909742007 0.5658364395 -0.6623997719 0.0134716110 0.7651932691 0.6436596744</rotation>
          <translation locked="true">-2499366.429888 -4256836.520906 4029217.965003</translation>
          <scale locked="true">12.354192447331316</scale>
        </transform>
      </component>
    </components>
    <cameras next_id="816" next_group_id="2">
      <group id="0" label="Group 1" type="folder">
        <camera id="0" sensor_id="0" component_id="0" label="/home/user/image_sets/0001/nadir/000001/000001-01/00/000001-01_00001.JPG">
          <transform>-0.99988146 -0.00872749 0.01268391 7.04224697 -0.00842932 0.99969124 0.02337453 7.05403071 -0.01288400 0.02326485 -0.99964631 0.43561421 0 0 0 1</transform>
          <orientation>1</orientation>
          <reference x="-120.417985477" y="39.411917469999999" z="2420.1089999999999" yaw="179.90000000000001" pitch="0.099999999999993788" roll="-0" enabled="true" rotation_enabled="false"/>
        </camera>
      </group>
    </cameras>
  </chunk>
</document>
"""


def camera_file(tmp_path: Path):
    # Parse the XML to validate
    root = ET.fromstring(CAMERA_XML)
    # Build tree and write to file
    tree = ET.ElementTree(root)
    filepath = tmp_path / "camera.xml"
    tree.write(filepath, encoding="utf-8", xml_declaration=True)
    return filepath


@pytest.fixture
def gradient():
    """Make an image with a white center gradated to a black outer ring."""

    # Coordinates
    size = 21
    center = size // 2
    x, y = np.meshgrid(np.arange(size), np.arange(size))
    # dist = np.maximum(np.abs(x - center), np.abs(y - center))
    dist = np.sqrt((x - center) ** 2 + (y - center) ** 2)

    # Normalize distance: 0 at center, 1 at max radius
    gradient = np.clip(1 - (dist / np.max(dist)), 0, 1)

    # Make RGB (same value in all channels for grayscale)
    image = np.stack([gradient] * 3, axis=2)

    # Scale to 0–255 uint8
    return (image * 255).astype(np.uint8)


class TestMetashapeCameraSetWarp:

    def test_instantiate(self, tmp_path):
        """Just check that the XML loads correctly"""
        cameras = MetashapeCameraSet(
            camera_file=camera_file(tmp_path),
            image_folder=tmp_path,
        )
        expected = {
            "b1": 0.5262024073,
            "b2": -0.3058334293,
            "k1": -0.0919367147,
            "k2": -0.0762807468,
            "k3": 0.1162639394,
            "k4": -0.0761413904,
            "p1": -0.0003134847,
            "p2": 0.0001164035,
        }
        for camera in cameras.cameras:
            for k, v in camera.distortion_params.items():
                assert np.isclose(v, expected[k])

    @pytest.mark.parametrize(
        "w2i,k1,relationship",
        (
            [True, 0, operator.eq],
            # If the radial term is positive, it means that in the equation
            # we have warp = ideal * 1+. That means to dewarp we will sample
            # from a wider area, and the image will darken
            [True, 1, operator.lt],
            # If the radial term is positive, it means that in the equation
            # we have warp = ideal * 1-. That means to dewarp we will sample
            # from a smaller area, and the image will lighten
            [True, -1, operator.gt],
            # The previous statements are inverted if you are warping in the
            # other direction.
            [False, 0, operator.eq],
            [False, 1, operator.gt],
            [False, -1, operator.lt],
        ),
    )
    def test_dewarp(self, tmp_path, gradient, w2i, k1, relationship):
        """
        Check that with simplified distortion params the image either gets
        lighter, darker, or stays the same.
        """
        cameras = MetashapeCameraSet(
            camera_file=camera_file(tmp_path),
            image_folder=tmp_path,
        )
        camera = cameras.cameras[0]
        # Greatly simplify the camera intrinsics
        camera.cx = 0
        camera.cy = 0
        camera.f = 100
        camera.image_height = gradient.shape[0]
        camera.image_width = gradient.shape[1]
        camera.image_size = gradient.shape[:2]
        # Simplify the camera parameters
        for param in ["b1", "b2", "k2", "k3", "k4", "p1", "p2"]:
            camera.distortion_params[param] = 0
        camera.distortion_params["k1"] = k1

        # Warp the image
        dewarped = cameras.dewarp_image(camera, gradient, warped_to_ideal=w2i)

        # Check whether the image on the average got lighter, darker, or
        # stayed the same
        assert relationship(dewarped.mean(), gradient.mean())
