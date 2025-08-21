import tempfile
from pathlib import Path

import numpy as np
import pytest
import pyvista as pv
import pyproj

from geograypher.cameras.cameras import PhotogrammetryCamera, PhotogrammetryCameraSet
from geograypher.meshes.derived_meshes import TexturedPhotogrammetryMeshPyTorch3dRendering


def make_simple_camera(position=(0, 0, 10)):
    """Create a simple camera looking down at the origin from above"""
    # Camera looking down the z-axis from position
    cam_to_world = np.array([
        [1, 0, 0, position[0]],
        [0, -1, 0, position[1]],
        [0, 0, -1, position[2]],
        [0, 0, 0, 1]
    ])
    return PhotogrammetryCamera(
        image_filename="/tmp/test_render.jpg",
        cam_to_world_transform=cam_to_world,
        f=1000,  # focal length in pixels
        cx=0,    # principal point offset x
        cy=0,    # principal point offset y
        image_width=200,
        image_height=200,
        local_to_epsg_4978_transform=np.eye(4),
    )


def create_flat_mesh_with_colored_points():
    """Create a flat mesh with 200x200 points (one per pixel) with some points colored red"""
    # Create a plane mesh with 200x200 points (one per pixel)
    plane = pv.Plane(
        center=(0, 0, 0),
        direction=(0, 0, 1),  # normal pointing up
        i_size=4,
        j_size=4,
        i_resolution=200 - 1,  # 200 points = 399 intervals
        j_resolution=200 - 1,
    )
    if plane is None:
        raise ValueError("Failed to create plane mesh")

    # Triangulate to ensure we have triangular faces
    plane = plane.triangulate()
    if plane is None:
        raise ValueError("Failed to create plane triangulation")

    # Create point colors: most points white (255, 255, 255), some points red (255, 0, 0)
    if plane is not None:
        n_points = plane.n_points
    else:
        raise ValueError("Failed to create plane mesh")
    point_colors = np.full((n_points, 3), fill_value=255, dtype=np.uint8)  # Start with all white

    # Make some points red in a pattern that will be visible
    # Create a checkerboard-like pattern or specific regions
    points = plane.points

    # Find points in specific regions to color red
    # red_indices = np.array([0, 1, 200, 201, 400, 401, 600, 601, 700, 201 * 25])
    # point_colors[red_indices] = [255, 0, 0]

    blue_indices = np.array([75, 76, 77, 300, 301, 302, 500, 501, 502])
    point_colors[blue_indices] = [0, 0, 255]

    return plane, point_colors


class TestTexturedPhotogrammetryMeshPyTorch3dRendering:

    def test_basic(self):

        # Create a simple flat mesh
        mesh, point_colors = create_flat_mesh_with_colored_points()

        # Create the TexturedPhotogrammetryMeshPyTorch3dRendering instance
        mesh_crs = pyproj.CRS.from_epsg(4978)  # ECEF

        textured_mesh = TexturedPhotogrammetryMeshPyTorch3dRendering(
            mesh=mesh,
            input_CRS=mesh_crs,
            texture=point_colors,
        )
        textured_mesh.save_mesh("/tmp/mesh.ply")

        # Create a camera positioned above the mesh
        camera = make_simple_camera(position=(0, 0, 10))

        # Render the mesh from this camera
        rendered_images = list(textured_mesh.render_flat(cameras=camera, return_camera=False))
        assert len(rendered_images) == 1
        rendered_image = rendered_images[0]

        # Check the rendered image properties
        assert rendered_image is not None
        # Type cast to help type checker understand this is an ndarray
        rendered_image = np.asarray(rendered_image)
        assert rendered_image.ndim == 3  # Should be (height, width, channels)
        assert rendered_image.shape[2] == 3

        print(rendered_image.dtype)
        from PIL import Image
        vis = Image.fromarray(rendered_image.astype(np.uint8))
        vis.save("/tmp/mesh.png")

        # Step 6: Check that some pixels are red and some are white
        # With point-based coloring, the rendered result will be interpolated
        # across faces based on the vertex colors
        height, width, channels = rendered_image.shape
        reshaped_image = rendered_image.reshape(-1, channels)

        # Remove NaN pixels (areas where no mesh was visible)
        valid_pixels = ~np.isnan(reshaped_image).any(axis=1)
        valid_pixel_colors = reshaped_image[valid_pixels]
