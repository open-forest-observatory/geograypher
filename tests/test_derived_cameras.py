import operator
import xml.etree.ElementTree as ET
from itertools import combinations, product
from pathlib import Path

import numpy as np
import pytest

from geograypher.cameras.derived_cameras import MetashapeCameraSet
from geograypher.constants import EARTH_CENTERED_EARTH_FIXED_CRS
from geograypher.meshes.meshes import TexturedPhotogrammetryMesh
from geograypher.utils.test_utils import (
    downward_view,
    make_simple_camera_set,
    make_simple_mesh,
)

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

    # Normalize circular distance: 0 at center, 1 at max radius
    dist = np.sqrt((x - center) ** 2 + (y - center) ** 2)
    gradient = np.clip(1 - (dist / np.max(dist)), 0, 1)

    # Make RGB (same value in all channels for grayscale)
    image = np.stack([gradient] * 3, axis=2)

    # Scale to 0–255 uint8
    return (image * 255).astype(np.uint8)


def simplify_camera(camera, image, delete=None):
    # Greatly simplify the camera intrinsics
    camera.cx = 0
    camera.cy = 0
    camera.f = 100
    camera._local_to_epsg_4978_transform = np.eye(4)
    camera.image_height = image.shape[0]
    camera.image_width = image.shape[1]
    camera.image_size = image.shape[:2]
    # Simplify the camera parameters
    for param in ["b1", "b2", "k1", "k2", "k3", "k4", "p1", "p2"]:
        camera.distortion_params[param] = 0
    if delete is not None:
        for key in delete:
            del camera.distortion_params[key]
    return camera


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
            [True, 10, operator.lt],
            # If the radial term is positive, it means that in the equation
            # we have warp = ideal * 1-. That means to dewarp we will sample
            # from a smaller area, and the image will lighten
            [True, -10, operator.gt],
            # The previous statements are inverted if you are warping in the
            # other direction.
            [False, 0, operator.eq],
            [False, 10, operator.gt],
            [False, -10, operator.lt],
        ),
    )
    @pytest.mark.parametrize("downsample", [1, 2])
    @pytest.mark.parametrize("grayscale", [True, False])
    def test_warp_dewarp_image(
        self, tmp_path, gradient, w2i, k1, relationship, downsample, grayscale
    ):
        """
        Check that with simplified distortion params the image either gets
        lighter, darker, or stays the same.
        """
        cameras = MetashapeCameraSet(
            camera_file=camera_file(tmp_path),
            image_folder=tmp_path,
        )
        camera = simplify_camera(cameras.cameras[0], gradient)
        camera.distortion_params["k1"] = k1

        if grayscale:
            gradient = gradient[:, :, 0]

        # Warp the image
        dewarped = cameras.warp_dewarp_image(
            camera, gradient, warped_to_ideal=w2i, inversion_downsample=downsample
        )

        # Check whether the image on the average got lighter, darker, or
        # stayed the same
        assert relationship(dewarped.mean(), gradient.mean())

    @pytest.mark.parametrize("k1", [1.0, 0.0, -1.0])
    @pytest.mark.parametrize("w2i", [True, False])
    @pytest.mark.parametrize("downsample", [1, 2])
    def test_mask_image(self, tmp_path, k1, w2i, downsample):
        """
        Check that if warping is applied to a mask image (a few integer classes)
        those classes are preserved.
        """

        image = np.ones((21, 21), dtype=np.uint8)
        image[:10] = 0
        image[:, 5:] = 2

        cameras = MetashapeCameraSet(
            camera_file=camera_file(tmp_path),
            image_folder=tmp_path,
        )
        camera = simplify_camera(cameras.cameras[0], image)
        camera.distortion_params["k1"] = k1

        # Warp the image
        dewarped = cameras.warp_dewarp_image(
            camera, image, warped_to_ideal=w2i, inversion_downsample=downsample
        )

        # Check that we kept our desired classes (no statements about position)
        assert sorted(np.unique(dewarped)) == [0, 1, 2]

    @pytest.mark.parametrize(
        "delete", combinations(["b1", "b2", "k1", "k2", "k3", "k4", "p1", "p2"], 2)
    )
    @pytest.mark.parametrize(
        "w2i,k1,relationship",
        (
            [True, 1, operator.lt],
            [True, -1, operator.gt],
            [False, 1, operator.gt],
            [False, -1, operator.lt],
        ),
    )
    @pytest.mark.parametrize("downsample", [1, 2])
    def test_dropped_parameters(
        self, tmp_path, gradient, delete, w2i, k1, relationship, downsample
    ):
        """
        Check that if distortion parameters are dropped we continue on. Arbitrarily
        drop combinations of two parameters, skipping k1.
        """

        cameras = MetashapeCameraSet(
            camera_file=camera_file(tmp_path),
            image_folder=tmp_path,
        )
        camera = simplify_camera(cameras.cameras[0], gradient, delete=delete)
        camera.distortion_params["k1"] = k1

        # Warp the image
        dewarped = cameras.warp_dewarp_image(
            camera, gradient, warped_to_ideal=w2i, inversion_downsample=downsample
        )

        # Check whether the image on the average got lighter, darker, or
        # stayed the same
        assert dewarped.shape == gradient.shape
        assert relationship(dewarped.mean(), gradient.mean())

    @pytest.mark.parametrize(
        "w2i,k1,relationship",
        (
            [True, 0, operator.eq],
            # If the radial term is positive, it means that in the equation
            # we have warp = ideal * 1+. That means to dewarp we will sample
            # from a wider area, and the image will darken
            [True, 0.5, operator.lt],
            # If the radial term is positive, it means that in the equation
            # we have warp = ideal * 1-. That means to dewarp we will sample
            # from a smaller area, and the image will lighten
            [True, -0.5, operator.gt],
            # The previous statements are inverted if you are warping in the
            # other direction.
            [False, 0, operator.eq],
            [False, 0.5, operator.gt],
            [False, -0.5, operator.lt],
        ),
    )
    @pytest.mark.parametrize("downsample", [1, 2])
    def test_warp_dewarp_pixels(self, tmp_path, w2i, k1, relationship, downsample):

        # Create a fake image size for our simplified camera
        fake = np.zeros((101, 101, 3), dtype=np.uint8)

        # Create a Metashape camera with simplified distortion
        cameras = MetashapeCameraSet(
            camera_file=camera_file(tmp_path),
            image_folder=tmp_path,
        )
        camera = simplify_camera(cameras.cameras[0], fake)
        camera.distortion_params["k1"] = k1

        # Choose pixels we want to sample (N, 2) in the 21x21 image
        pixels = np.array(
            [
                [20, 20],
                [20, 50],
                [20, 80],
                [50, 20],
                [50, 80],
                [80, 20],
                [80, 50],
                [80, 80],
            ]
        )
        center = np.mean([[0, 0], fake.shape[:2]], axis=0).astype(int)

        # Warp the pixels
        dewarped = cameras.warp_dewarp_pixels(
            camera, pixels, warped_to_ideal=w2i, inversion_downsample=downsample
        )

        # Basic size and type checks
        assert isinstance(dewarped, np.ndarray)
        assert dewarped.shape == pixels.shape
        assert dewarped.dtype == int

        # Check whether the pixels regressed towards the center or away from
        # the center, as appropriate
        original = np.linalg.norm(pixels - center, axis=1)
        altered = np.linalg.norm(dewarped - center, axis=1)
        assert relationship(altered, original).all()


class TestPix2Face:

    def test_simple_camera(self):
        """Warp/dewarp should fail if not given a Metashape camera."""

        # Create simple mesh and camera
        mesh, point_colors = make_simple_mesh(pixels=[], color=None)
        textured_mesh = TexturedPhotogrammetryMesh(
            mesh=mesh,
            input_CRS=EARTH_CENTERED_EARTH_FIXED_CRS,
            texture=point_colors,
        )
        cameras = make_simple_camera_set()

        # Render the mesh from these cameras
        with pytest.raises(NotImplementedError):
            textured_mesh.pix2face(
                cameras=cameras,
                cache_folder=None,
                distortion_set=cameras,
                apply_distortion=True,
            )

    @pytest.mark.parametrize("render_img_scale", [0.5, 0.7, 0.9, 1.0])
    def test_dewarp_pix2face(self, tmp_path, render_img_scale):
        """
        Test that when we park a camera right over a mesh and call pix2face
        the result gets warped (when set).
        """

        # Load a simple 2x2 mesh with no colors
        mesh, point_colors = make_simple_mesh(pixels=[], color=None)
        textured_mesh = TexturedPhotogrammetryMesh(
            mesh=mesh,
            input_CRS=EARTH_CENTERED_EARTH_FIXED_CRS,
            texture=point_colors,
        )

        # Load a metashape camera set and then clean it up so it is looking
        # directly at the simple mesh.
        cameras = MetashapeCameraSet(
            camera_file=camera_file(tmp_path),
            image_folder=tmp_path,
        )

        # Pick an arbitrary sensor size and K1 such that the warping looks somewhat
        # like our cameras do in practice (corners pulled slightly in).
        # Note that simplify_camera zeroes out the other distortion parameters.
        # Note that the choice of K1 is tied to the sensor size by the radius.
        sensor = 2**8 + 1
        camera = simplify_camera(cameras.cameras[0], image=np.ones((sensor, sensor, 3)))
        camera.distortion_params["k1"] = -0.05

        # In addition to simplifying the camera, we need to make sure the set transform
        # is also removed
        cameras._local_to_epsg_4978_transform = np.eye(4)

        # Adjust the camera transform so we are looking down with the mesh face in view
        HT = downward_view(
            scene_width=4,  # Known width of the simple mesh in X/Y
            focal=cameras.cameras[0].f,
            sensor_width=sensor,
        )
        cameras.cameras[0].cam_to_world_transform = HT
        cameras.cameras[0].world_to_cam_transform = np.linalg.inv(HT)

        # Calculate the ideal (plain render) and warped (plain render → apply
        # distortion parameters) images
        kwargs = {
            "cameras": cameras,
            "cache_folder": None,
            "distortion_set": cameras,
            "render_img_scale": render_img_scale,
        }
        ideal = textured_mesh.pix2face(**kwargs, apply_distortion=False)
        assert len(ideal) == 1
        ideal = ideal[0]
        warped = textured_mesh.pix2face(**kwargs, apply_distortion=True)
        assert len(warped) == 1
        warped = warped[0]

        # Make some basic assumptions about shape and type
        scaled_sensor = int(sensor * render_img_scale)
        for image in [ideal, warped]:
            assert isinstance(image, np.ndarray)
            assert image.dtype == np.int64
            assert image.shape == (scaled_sensor, scaled_sensor)
            # The "invalid" marker
            assert image.min() >= -1
            # Don't make explicit statements about the max visible face index
            # (because any individual face might not align to a pixel), but
            # it should be near the number of faces
            assert image.max() < mesh.n_faces
            assert image.max() > 0.95 * mesh.n_faces

        # Broad statement: when the image is warped it should pull the corners
        # in and cause the corners to be invalid
        for corner in product([slice(None, 10), slice(-10, None)], repeat=2):
            assert len(np.unique(ideal[corner])) > 1
            assert np.all(warped[corner] == -1)
