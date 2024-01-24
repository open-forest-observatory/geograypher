import typing

import numpy as np


def create_ramped_weighting(
    rectangle_shape: typing.Tuple[int, int], ramp_dist_frac: float
) -> np.ndarray:
    """Create a ramped weighting that is higher toward the center with a max value of 1 at a fraction from the edge

    Args:
        rectangle_shape (typing.Tuple[int, int]): Size of rectangle to create a mask for
        ramp_dist_frac (float): Portions at least this far from an edge will have full weight

    Returns:
        np.ndarray: An array representing the weighting from 0-1
    """
    i_ramp = np.clip(np.linspace(0, 1 / ramp_dist_frac, num=rectangle_shape[0]), 0, 1)
    j_ramp = np.clip(np.linspace(0, 1 / ramp_dist_frac, num=rectangle_shape[1]), 0, 1)

    i_ramp = np.minimum(i_ramp, np.flip(i_ramp))
    j_ramp = np.minimum(j_ramp, np.flip(j_ramp))

    i_ramp = np.expand_dims(i_ramp, 1)
    j_ramp = np.expand_dims(j_ramp, 0)

    ramped_weighting = np.minimum(i_ramp, j_ramp)
    return ramped_weighting


def compute_3D_triangle_area_vectorized(corners: np.ndarray, return_z_proj_area=True):
    """_summary_

    Args:
        corners (np.ndarray): (n_faces, n)
        return_z_proj_area (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    A, B, C = corners
    # https://math.stackexchange.com/questions/2152754/calculate-3d-triangle-area-by-determinant
    u = B - A
    v = C - A

    # Save for future computation
    u0v1_min_u1v0 = u[0] * v[1] - u[1] * v[0]
    area = (
        1
        / 2
        * np.sqrt(
            np.power(u[1] * v[2] - u[2] * v[1], 2)
            + np.power(u[2] * v[0] - u[0] * v[2], 2)
            + np.power(u0v1_min_u1v0, 2)
        )
    )

    if return_z_proj_area:
        area_z_proj = np.abs(u0v1_min_u1v0) / 2
        return area, area_z_proj

    return area


def compute_3D_triangle_area(corners, return_z_proj_area=True):
    A, B, C = corners
    # https://math.stackexchange.com/questions/2152754/calculate-3d-triangle-area-by-determinant
    u = B - A
    v = C - A

    # Save for future computation
    u0v1_min_u1v0 = u[0] * v[1] - u[1] * v[0]
    area = (
        1
        / 2
        * np.sqrt(
            np.power(u[1] * v[2] - u[2] * v[1], 2)
            + np.power(u[2] * v[0] - u[0] * v[2], 2)
            + np.power(u0v1_min_u1v0, 2)
        )
    )

    if return_z_proj_area:
        area_z_proj = np.abs(u0v1_min_u1v0) / 2
        return area, area_z_proj

    return area
