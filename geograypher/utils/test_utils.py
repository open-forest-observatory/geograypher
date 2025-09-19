from itertools import product
from typing import List, Tuple

import numpy as np
import pyvista as pv

from geograypher.cameras.cameras import PhotogrammetryCamera, PhotogrammetryCameraSet


def make_simple_camera_set() -> PhotogrammetryCameraSet:
    """
    Create a simple camera looking down at the origin from above, spaced to line up
    perfectly with the simple mesh.
    """

    # Focal length in pixels
    focal = 100
    # Sensor width/height in pixels
    sensor = 200

    return PhotogrammetryCameraSet(
        cameras=[
            PhotogrammetryCamera(
                image_filename=None,
                cam_to_world_transform=downward_view(
                    scene_width=4,  # Known width of the simple mesh in X/Y
                    focal=focal,
                    sensor_width=sensor,
                ),
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


def downward_view(scene_width, focal, sensor_width):
    """
    The equation is just a ratio of triangles
    Camera distance (m) = Scene width (m) * focal length (px) / sensor width (px)

       |.
       |   .               .|
    ↑  |       .        .   | ↑
    sw |  ←cd→   [cam]  ←f→ | sensor w
    ↓  |       .        .   | ↓
       |   .               .|
       |.

    Note that the orientation is set so the camera is looking down. If you imagine
    rotating a right-hand reference frame by 180 so Z (out of the camera) points
    down, you end up with Y pointing along -Y and Z pointing along -Z.
    """
    return np.array(
        [
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, scene_width * focal / sensor_width],
            [0, 0, 0, 1],
        ]
    )


def make_simple_mesh(
    pixels: np.ndarray, color: List[int], background: int = 50, buffer: int = 0
) -> Tuple[pv.PolyData, np.ndarray]:
    """
    Create a flat mesh with the given pixels colored. Designed to line up
    with the simple camera so that 1 interval between points = 1 pixel.

    Arguments:
        pixels (ndarray): (Q, 2) array of (i, j) pixel values we want to color
        color (List[int]): 3-element 0-255 color to set the pixels to
        background (int):
        buffer (int): Number of pixels to offset around each pixel with the
            same color

    Returns: Two element tuple of
        [0] pyvista mesh of the plane
        [1] (M, 3) color vector for each point in the mesh
    """

    # Define the number of pixels we want to support. 200 intervals (pixels) =
    # 201 points in the mesh.
    N = 201

    plane = pv.Plane(
        center=(0, 0, 0),
        direction=(0, 0, 1),  # normal pointing up
        i_size=4,  # From -2 to 2
        j_size=4,  # From -2 to 2
        i_resolution=N - 1,
        j_resolution=N - 1,
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
        pixel_idx(point_colors, *pixel, stride=N, color=color, buffer=buffer)

    ##########################################################################
    # For visual debugging purposes, uncomment this code, it will color the
    # edges of the mesh in bright colors.
    # point_colors[0] = [255, 0, 0]
    # point_colors[1] = [0, 0, 255]
    # point_colors[:2*N] = [255, 0, 0]
    # point_colors[-2*N:] = [0, 255, 0]
    # point_colors[0::N] = [0, 0, 255]
    # point_colors[1::N] = [0, 0, 255]
    # point_colors[N-2::N] = [0, 255, 255]
    # point_colors[N-1::N] = [0, 255, 255]
    ##########################################################################

    return plane, point_colors


def pixel_idx(
    vector: np.ndarray, i: int, j: int, stride: int, color: List[int], buffer: int = 0
) -> None:
    """
    Helper to turn unflattened ij pixels into the color vector around
    that mesh area. Note that we have to invert the i dimension b/c of
    how plane texturing is applied vs. how images are rendered. This
    way (i, j) will correspond to (i, j) in the image.

    Arguments:
        vector (ndarray): Shape (M*N,) array, flattened (M, N) image
        i, j (int): row, column location we want to set a color
        stride: N, a.k.a. the number of columns in the unflattened image
        color (List[int]): 3-element 0-255 color to set the pixel to
        buffet (int): Number of pixels to offset around each pixel with the
            same color

    Returns: None, vector is modified in place
    """

    # For robustness purposes we may want to color more than 1 pixel
    spread = range(-buffer, 2 + buffer)

    for di, dj in product(spread, spread):
        vector[(stride - i - di) * stride + (j + dj)] = color
