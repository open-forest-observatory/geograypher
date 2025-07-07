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


def compute_approximate_ray_intersection(
    a0: np.ndarray,
    a1: np.ndarray,
    b0: np.ndarray,
    b1: np.ndarray,
    clamp: bool = False,
    plotter=None,
):
    """
    Given a single line segment defined by 3D numpy.array points (a0, a1) and N line segments
    defined by (b0, b1) (each (N, 3)), return the closest points on each segment and their
    distances. If clamp is True, then respect the line segment ends. If clamp is False,
    then use the infinite rays.

    Args:
        a0 (np.ndarray): Start point of the first segment (shape: (3,)).
        a1 (np.ndarray): End point of the first segment (shape: (3,)).
        b0 (np.ndarray): Start points of the second segments (shape: (N, 3)).
        b1 (np.ndarray): End points of the second segments (shape: (N, 3)).
        clamp (bool, optional): If True, the closest points are clamped to the segment
            endpoints. If False, the closest points may be anywhere along the infinite
            lines.
        plotter (pyvista.Plotter, optional): If provided, visualizes the segments
            and closest points using pyvista.

    Returns:
        pA (np.ndarray): Closest points on the first segment or line (shape: (N, 3)),
            or None if segments are parallel and overlap.
        pB (np.ndarray): Closest points on the second segments or lines (shape: (N, 3)),
            or None if segments are parallel and overlap.
        dists (np.ndarray): The minimum distances between the two segments or lines (shape: (N,)).
    """

    # Normalize both the A vector and all B vectors
    A = a1 - a0
    magA = np.linalg.norm(A)
    _A = A / magA
    B = b1 - b0
    magB = np.linalg.norm(B, axis=1)
    _B = B / magB[:, None]

    cross = np.cross(_A, _B)
    denom = np.linalg.norm(cross, axis=1) ** 2

    # Where the lines criss-cross (denom > 0): calculate the projected closest points
    t = b0 - a0
    detA = np.einsum("ij,ij->i", np.cross(t, _B), cross)
    detB = np.einsum("ij,ij->i", np.cross(t, np.broadcast_to(_A, _B.shape)), cross)

    # Where the denom is 0, for this division step replace it with 1
    denom_safe = np.where(denom == 0, 1, denom)
    # Scale vectors that stretch along the A and B vectors. If the scale is
    # between 0 and the magnitude of that vector then the projected point is
    # within the line segment
    t0 = detA / denom_safe
    t1 = detB / denom_safe

    # Projected closest point on segments A and B
    pA = a0 + t0[:, None] * _A
    pB = b0 + t1[:, None] * _B
    # Clamp projections
    if clamp:

        t0_clamped = np.clip(t0, 0, magA)
        pA = a0 + t0_clamped[:, None] * _A

        t1_clamped = np.clip(t1, 0, magB)
        pB = b0 + t1_clamped[:, None] * _B

        # Check if anything needs recomputing due to clamping (oob = out of bounds)
        oob_A = (t0 < 0) | (t0 > magA)
        oob_B = (t1 < 0) | (t1 > magB)

        # Recompute pB where A is clamped
        if np.any(oob_A):
            dot = np.einsum("ij,ij->i", pA[oob_A] - b0[oob_A], _B[oob_A])
            pB[oob_A] = b0[oob_A] + np.clip(dot, 0, magB[oob_A])[:, None] * _B[oob_A]

        # Recompute pA where B is clamped
        if np.any(oob_B):
            dot = np.einsum("ij,j->i", pB[oob_B] - a0, _A)
            pA[oob_B] = a0 + np.clip(dot, 0, magA)[:, None] * _A

    # Handle exact parallel case by substituting results
    parallel = denom == 0
    if np.any(parallel):

        d0 = np.einsum("j,ij->i", _A, b0 - a0)
        if clamp:

            d1 = np.einsum("j,ij->i", _A, b1 - a0)

            before = (d0 <= 0) & (d1 <= 0) & parallel
            after = (d0 >= magA) & (d1 >= magA) & parallel
            middle = parallel & ~(before | after)

            if np.any(before):
                pA[before] = a0
                pB[before] = np.where(
                    np.abs(d0[before]) < np.abs(d1[before])[:, None],
                    b0[before],
                    b1[before],
                )

            if np.any(after):
                pA[after] = a1
                pB[after] = np.where(
                    np.abs(d0[after]) < np.abs(d1[after])[:, None],
                    b0[after],
                    b1[after],
                )

            if np.any(middle):
                t_mid = np.clip(d0[middle], 0, magA)
                pA[middle] = a0 + t_mid[:, None] * _A
                # Vector from A to B starting point
                a2b = b0[middle] - pA[middle]
                # Component along A
                alongA = np.einsum("ij,j->i", a2b, _A)[:, None] * _A
                # Remove the parallel component so we are left only with perpendicular
                perpendicular = a2b - alongA
                pB[middle] = pA[middle] + perpendicular

        else:
            pA[parallel] = a0 + d0[:, None] * _A
            pB[parallel] = b0

    if plotter is not None:
        raise NotImplementedError()
        # points = np.vstack([a0, pA, a1, b1, pB, b0])
        # plotter.add_lines(points)
        # plotter.add_points(points)
        # plotter.background_color = "black"
        # plotter.show()

    return pA, pB, np.linalg.norm(pA - pB, axis=1)


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

    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return x


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


def intersection_average(starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """
    Given arrays of line segment start and end points, compute the average of the closest
    intersection points between all pairs of segments.

    Args:
        starts (np.ndarray): (N, 3) array of segment start points
        ends (np.ndarray): (N, 3) array of segment end points

    Returns:
        np.ndarray: (3,) array, the average intersection point
    """
    N = starts.shape[0]
    closest_points = []
    for i in range(N - 1):
        a0, a1 = starts[i], ends[i]
        b0, b1 = starts[i + 1 : N], ends[i + 1 : N]
        pA, pB, _ = compute_approximate_ray_intersection(a0, a1, b0, b1, clamp=True)
        closest_points.append(pA)
        closest_points.append(pB)
    if closest_points:
        return np.mean(np.vstack(closest_points), axis=0)
    else:
        # If all are None, return the average of all start and end points
        all_points = np.concatenate([starts, ends], axis=0)
        return np.mean(all_points, axis=0)
