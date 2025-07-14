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
    a0: np.array,
    a1: np.array,
    b0: np.array,
    b1: np.array,
    clamp: bool = False,
    plotter=None,
):
    """
    Given two line segments defined by 3D numpy.array pairs (a0, a1, b0, b1), return the
    closest points on each segment and their distance. If clamp is True, then respect
    the line segment ends. If clamp is False, then use the infinite rays.

    Based on https://stackoverflow.com/questions/2824478/shortest-distance-between-two-line-segments

    Args:
        a0 (np.ndarray): Start point of the first segment (shape: (3,)).
        a1 (np.ndarray): End point of the first segment (shape: (3,)).
        b0 (np.ndarray): Start point of the second segment (shape: (3,)).
        b1 (np.ndarray): End point of the second segment (shape: (3,)).
        clamp (bool, optional): If True, the closest points are clamped to the segment
            endpoints. If False, the closest points may be anywhere along the infinite
            lines.
        plotter (pyvista.Plotter, optional): If provided, visualizes the segments
            and closest points using pyvista.

    Returns:
        pA (np.ndarray): Closest point on the first segment or line (shape: (3,)),
            or None if segments are parallel and overlap.
        pB (np.ndarray): Closest point on the second segment or line (shape: (3,)),
            or None if segments are parallel and overlap.
        dist (float): The minimum distance between the two segments or lines.
    """

    # Calculate vectors, normalized vectors, and denominator
    A = a1 - a0
    B = b1 - b0
    magA = np.linalg.norm(A)
    magB = np.linalg.norm(B)
    # Normalized vectors
    _A = A / magA
    _B = B / magB
    cross = np.cross(_A, _B)
    # Denom is the area of the parallelogram formed by the A and B unit vectors
    # (norm of the cross product), squared. As that area goes to zero, it
    # means that the unit vectors are aligned.
    denom = np.linalg.norm(cross) ** 2

    # If lines are parallel (denom=0) test if lines overlap. If they don't
    # overlap then there is a closest point solution. If they do overlap,
    # there are infinite closest positions, but there is a closest distance
    if denom == 0:
        d0 = np.dot(_A, (b0 - a0))

        # Overlap only possible with clamping
        if clamp:
            d1 = np.dot(_A, (b1 - a0))

            # Is segment B before A?
            if (0 >= d0) and (0 >= d1):
                if np.absolute(d0) < np.absolute(d1):
                    return a0, b0, np.linalg.norm(a0 - b0)
                return a0, b1, np.linalg.norm(a0 - b1)

            # Is segment B after A?
            elif (magA <= d0) and (magA <= d1):
                if np.absolute(d0) < np.absolute(d1):
                    return a1, b0, np.linalg.norm(a1 - b0)
                return a1, b1, np.linalg.norm(a1 - b1)

        # Segments overlap, return distance between parallel segments.
        # Closest point is meaningless on parallel lines.
        return None, None, np.linalg.norm((a0 + (d0 * _A)) - b0)

    # Lines criss-cross: Calculate the projected closest points
    t = b0 - a0
    detA = np.linalg.det([t, _B, cross])
    detB = np.linalg.det([t, _A, cross])

    # Scale vectors that stretch along the A and B vectors. If
    # the scale is between 0 and the magnitude of that vector
    # then the projected point is within the line segment
    t0 = detA / denom
    t1 = detB / denom

    # Projected closest point on rays A and B
    pA = a0 + (_A * t0)
    pB = b0 + (_B * t1)

    # Clamp projections
    if clamp:
        if t0 < 0:
            pA = a0
        elif t0 > magA:
            pA = a1

        if t1 < 0:
            pB = b0
        elif t1 > magB:
            pB = b1

        # Clamp projection A
        if (t0 < 0) or (t0 > magA):
            dot = np.dot(_B, (pA - b0))
            if dot < 0:
                dot = 0
            elif dot > magB:
                dot = magB
            pB = b0 + (_B * dot)

        # Clamp projection B
        if (t1 < 0) or (t1 > magB):
            dot = np.dot(_A, (pB - a0))
            if dot < 0:
                dot = 0
            elif dot > magA:
                dot = magA
            pA = a0 + (_A * dot)

    if plotter is not None:
        points = np.vstack([a0, pA, a1, b1, pB, b0])
        plotter.add_lines(points)
        plotter.add_points(points)
        plotter.background_color = "black"
        plotter.show()

    return pA, pB, np.linalg.norm(pA - pB)


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
    for i in range(N):
        for j in range(i + 1, N):
            a0, a1 = starts[i], ends[i]
            b0, b1 = starts[j], ends[j]
            pA, pB, _ = compute_approximate_ray_intersection(a0, a1, b0, b1, clamp=True)
            if pA is not None and pB is not None:
                closest_points.append(pA)
                closest_points.append(pB)
    if len(closest_points) > 0:
        return np.mean(np.stack(closest_points, axis=0), axis=0)
    else:
        # If all are None, return the average of all start and end points
        all_points = np.concatenate([starts, ends], axis=0)
        return np.mean(all_points, axis=0)
