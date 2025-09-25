import tempfile
from pathlib import Path

import numpy as np
import pyproj
import pytest

from geograypher.constants import EARTH_CENTERED_EARTH_FIXED_CRS
from geograypher.meshes.derived_meshes import (
    TexturedPhotogrammetryMeshPyTorch3dRendering,
)
from geograypher.meshes.meshes import TexturedPhotogrammetryMesh
from geograypher.utils.test_utils import make_simple_camera_set, make_simple_mesh


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
    cameras = make_simple_camera_set()

    # Render the mesh from this camera
    renders = list(
        textured_mesh.render_flat(
            cameras=cameras,
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
