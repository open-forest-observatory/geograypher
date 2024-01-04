import typing

import numpy as np
import pyvista as pv


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


def compute_approximate_ray_intersection(
    A: np.array, a: np.array, B: np.array, b: np.array, vis=False
):
    # https://palitri.com/vault/stuff/maths/Rays%18closest%20point.pdf
    c = B - A

    aa = np.dot(a, a)
    ab = np.dot(a, b)
    ac = np.dot(a, c)
    bb = np.dot(b, b)
    bc = np.dot(b, c)

    denominator = aa * bb - ab * ab

    a_scaler = (-ab * bc + ac * bb) / denominator
    b_scaler = (ab * ac - bc * aa) / denominator

    D = A + a_scaler * a
    E = B + b_scaler * b

    dist = np.linalg.norm(D - E)
    # Check that it's in front of the camera
    # TODO I'm not sure if this will handle co-linear cases, probably should check for that
    valid = a_scaler > -1 and b_scaler > 0

    if vis:
        points = np.vstack([A, D, B, E, D, E])

        plotter = pv.Plotter()
        plotter.add_lines(points)
        plotter.add_points(points)
        plotter.background_color = "black"
        plotter.show()

    return dist, valid


def triangulate_rays_lstsq(starts, directions):
    # https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/geometry/representation/ray.py#L284-L371

    # Ensure that we have unit vectors specifying the direction of the ray

    # Build a cross product matrix for each direction, this is the left side
    # The right side is the dot product between this and the start ray

    As = []
    bs = []
    # TODO build this all at once rather than iteratively
    for start, direction in zip(starts, directions):
        x, y, z = direction
        # https://math.stackexchange.com/questions/3764426/matrix-vector-multiplication-cross-product-problem
        cross_matrix = np.array(
            [
                [0, -z, y],
                [z, 0, -x],
                [-y, x, 0],
            ]
        )
        As.append(cross_matrix)
        b = cross_matrix @ start
        bs.append(b)

    A = np.concatenate(As, axis=0)
    b = np.concatenate(bs, axis=0)

    x, _, _, _ = np.linalg.lstsq(A, b)
    return x
