import tempfile
from itertools import product
from pathlib import Path

import numpy as np
import pyproj
import pytest
import pyvista as pv

from geograypher.cameras.cameras import PhotogrammetryCamera, PhotogrammetryCameraSet
from geograypher.constants import EARTH_CENTERED_EARTH_FIXED_CRS
from geograypher.meshes.derived_meshes import (
    TexturedPhotogrammetryMeshPyTorch3dRendering,
)
from geograypher.meshes.meshes import TexturedPhotogrammetryMesh


def make_simple_camera():
    """
    Create a simple camera looking down at the origin from above, spaced to line up
    perfectly with the simple mesh.
    """

    # Focal length in pixels
    focal = 100
    # Sensor width/height in pixels
    sensor = 200

    # Known width of the simple mesh in the X/Y plane
    mesh_width = 4

    # The equation is just a ratio of triangles
    # Camera distance (m) = Scene width (m) * focal length (px) / sensor width (px)
    #
    #    |.
    #    |   .               .|
    # ↑  |       .        .   | ↑
    # sw |  ←cd→   [cam]  ←f→ | sensor w
    # ↓  |       .        .   | ↓
    #    |   .               .|
    #    |.
    #
    # Note that the orientation is set so the camera is looking down. If you imagine
    # rotating a right-hand reference frame by 180 so Z (out of the camera) points
    # down, you end up with Y pointing along -Y and Z pointing along -Z.
    cam_to_world = np.array(
        [
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, mesh_width * focal / sensor],
            [0, 0, 0, 1],
        ]
    )
    return PhotogrammetryCameraSet(
        cameras=[
            PhotogrammetryCamera(
                image_filename=None,
                cam_to_world_transform=cam_to_world,
                f=focal,
                cx=0,
                cy=0,
                image_width=sensor,
                image_height=sensor,
                local_to_epsg_4978_transform=np.eye(4),
            )
        ],
        local_to_epsg_4978_transform=np.eye(4),
    )


def pixel_idx(vector, i, j, stride, color, buffer=0):
    """
    Helper to turn unflattened ij pixels into the color vector around
    that mesh area. Note that we have to invert the i dimension b/c of
    how plane texturing is applied vs. how images are rendered. This
    way (i, j) will correspond to (i, j) in the image.
    """

    # For robustness purposes we may want to color more than 1 pixel
    spread = range(-buffer, 2 + buffer)

    for di, dj in product(spread, spread):
        vector[(stride - i - di) * stride + (j + dj)] = color


def make_simple_mesh(pixels, color, background=50, buffer=0):
    """
    Create a flat mesh with the given pixels colored. Designed to line up
    with the simple camera so that 1 interval between points = 1 pixel.
    """

    # Define the number of pixels we want to support. 200 intervals (pixels) =
    # 201 points in the mesh.
    pw = 201

    plane = pv.Plane(
        center=(0, 0, 0),
        direction=(0, 0, 1),  # normal pointing up
        i_size=4,  # From -2 to 2
        j_size=4,  # From -2 to 2
        i_resolution=pw - 1,
        j_resolution=pw - 1,
    )
    if plane is None:
        raise ValueError("Failed to create plane mesh")

    # Triangulate to ensure we have triangular faces
    plane = plane.triangulate()
    if plane is None:
        raise ValueError("Failed to create plane triangulation")

    # Create point colors
    n_points = plane.n_points
    point_colors = np.full((n_points, 3), fill_value=background, dtype=np.uint8)

    # Fill them in pixel by pixel
    for pixel in pixels:
        pixel_idx(point_colors, *pixel, stride=pw, color=color, buffer=buffer)

    ##########################################################################
    # For visual debugging purposes, uncomment this code, it will color the
    # edges of the image.
    # point_colors[0] = [255, 0, 0]
    # point_colors[1] = [0, 0, 255]
    # point_colors[:2*pw] = [255, 0, 0]
    # point_colors[-2*pw:] = [0, 255, 0]
    # point_colors[0::pw] = [0, 0, 255]
    # point_colors[1::pw] = [0, 0, 255]
    # point_colors[pw-2::pw] = [0, 255, 255]
    # point_colors[pw-1::pw] = [0, 255, 255]
    ##########################################################################

    return plane, point_colors


@pytest.mark.parametrize(
    "meshclass",
    [
        TexturedPhotogrammetryMesh,
        # TexturedPhotogrammetryMeshPyTorch3dRendering,
    ],
)
def test_perspective_camera(meshclass):

    fill_pixels = np.array([[10, 20], [15, 190], [195, 5], [50, 100], [150, 120]])
    empty_pixels = np.array([[30, 40], [160, 180], [120, 40], [100, 150], [180, 100]])

    # Create a simple flat mesh
    mesh, point_colors = make_simple_mesh(
        pixels=fill_pixels,
        color=[255, 0, 0],
        background=80,
        buffer=1,
    )

    # Create the textured mesh
    textured_mesh = meshclass(
        mesh=mesh,
        input_CRS=EARTH_CENTERED_EARTH_FIXED_CRS,
        texture=point_colors,
    )
    ######################################################################
    # For visual debugging purposes, uncomment this code
    # textured_mesh.save_mesh("/tmp/mesh.ply")
    ######################################################################

    # Create a camera positioned above the mesh
    camera = make_simple_camera()

    # Render the mesh from this camera
    renders = list(
        textured_mesh.render_flat(
            cameras=camera,
            return_camera=False,
            apply_distortion=False,
        )
    )
    assert len(renders) == 1
    render = renders[0]

    # Check the rendered image properties
    assert render is not None
    render = np.asarray(render)
    # Should be (height, width, channels)
    assert render.ndim == 3
    assert render.shape[2] == 3

    ######################################################################
    # For visual debugging purposes, uncomment this code
    # from PIL import Image
    # Image.fromarray(render.astype(np.uint8)).save("/tmp/mesh.png")
    ######################################################################

    # Check the expected pixel positions
    assert np.allclose(render[fill_pixels[:, 0], fill_pixels[:, 1]], [255, 0, 0])
    assert np.allclose(render[empty_pixels[:, 0], empty_pixels[:, 1]], [80, 80, 80])
