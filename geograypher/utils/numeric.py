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


def compute_approximate_ray_intersections(
    a0: np.ndarray, a1: np.ndarray, b0: np.ndarray, b1: np.ndarray, clamp: bool = False
) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute closest points and distances between N line segments a0->a1 and
    b0->b1. Returns (N, N, 3), (N, N, 3), (N, N). If clamp is True, then respect
    the line segment ends. If clamp is False, then use the infinite rays.

    Based on https://stackoverflow.com/questions/2824478/shortest-distance-between-two-line-segments

    Args:
        a0 (np.ndarray): Start points of the first segments (N, 3).
        a1 (np.ndarray): End points of the first segments (N, 3).
        b0 (np.ndarray): Start points of the second segments (N, 3).
        b1 (np.ndarray): End points of the second segments (N, 3).
        clamp (bool, optional): If True, the closest points are clamped to the
            segment endpoints. If False, the closest points may be anywhere
            along the infinite rays.

    Returns:
        pA (np.ndarray): Closest point on the A segments (axis 0) compared to
            each of the B segments (axis 1) (N, N, 3). For example, the closest
            point between A[5] and B[2] at pA[5, 2]
        pB (np.ndarray): Closest point on the B segments (axis 1) compared to
            each of the A segments (axis 1) (N, N, 3)
        dist (np.ndarray): The minimum distance between the A (axis 0) and B
            (axis 1) segments.
    """

    # (N, 3) vectors, representing the A and B line segments
    A = a1 - a0
    B = b1 - b0
    # (N,) distances, representing the length of each A and B vector
    magA = np.linalg.norm(A, axis=1)
    magB = np.linalg.norm(B, axis=1)
    # (N, 3) vectors, representing the A and B unit vectors
    _A = A / magA[:, None]
    _B = B / magB[:, None]

    # Expand the A vectors to (N, 1, 3) and the B vectors to (1, N, 3) so that
    # they project together to an (N, N, 3) matrix later
    a0_exp = a0[:, None, :]
    b0_exp = b0[None, :, :]
    _A_exp = _A[:, None, :]
    _B_exp = _B[None, :, :]

    # (N, N, 3) cross product. The cross product of unit vectors calculates the
    # area of the parallelogram formed by the unit vectors, a larger value
    # indicates lines that are more orthogonal
    cross = np.cross(_A_exp, _B_exp)
    # (N, N) representing the squared area of the unit vector parallelograms
    denom = np.linalg.norm(cross, axis=2) ** 2

    # (N, N, 3) matrix, representing the vector from the start of each A
    # segment to the start of each B segment
    t = b0_exp - a0_exp

    # (N, N) matrix, calculate the determinant by multiplying all axes (ijk)
    # and then sum across k, leaving only an ij matrix. The determinant
    # represents how a given transform scales space
    detA = np.einsum("ijk,ijk->ij", np.cross(t, _B_exp), cross)
    detB = np.einsum("ijk,ijk->ij", np.cross(t, _A_exp), cross)

    # Make the denom safe for division by (temporarily) replacing 0 values
    # with 1. Later we check for 0 locations and fix them
    denom_safe = np.where(denom == 0, 1, denom)
    # t0 and t1 are (N, N) matrices representing how far along A the closest
    # point is (t0) where 0 is at a0 and 1 is at a1. B and t1 are the same.
    t0 = detA / denom_safe
    t1 = detB / denom_safe

    if clamp:
        # (N, N) matrix where t0 and t1 are clipped to the vector length
        t0_clamped = np.clip(t0, 0, magA[:, None])
        t1_clamped = np.clip(t1, 0, magB[None, :])

        # (N, N, 3) matrix where we start at each vector starting point (a0
        # and b0 expanded) then travel along them to the closest clamped point
        pA = a0_exp + t0_clamped[:, :, None] * _A_exp
        pB = b0_exp + t1_clamped[:, :, None] * _B_exp

        # Check whether the original scale values where out of bounds
        oob_A = (t0 < 0) | (t0 > magA[:, None])
        oob_B = (t1 < 0) | (t1 > magB[None, :])

        # Broadcast a0 and _A to (N, N, 3), similar for b0/_B
        a0_bcast = np.broadcast_to(a0[:, None, :], pA.shape)
        _A_bcast = np.broadcast_to(_A[:, None, :], pA.shape)
        b0_bcast = np.broadcast_to(b0[None, :, :], pB.shape)
        _B_bcast = np.broadcast_to(_B[None, :, :], pB.shape)

        # If any of the scale vectors were clipped, we may have to adjust the
        # closest point on the opposite vector. For example if the closest
        # point on an A vector was clipped, the corresponding B point may need
        # to update
        if np.any(oob_A):
            # Get the projection of the new A point onto the B vectors
            dot = np.einsum("ijk,ijk->ij", pA - b0_bcast, _B_bcast)
            # Get a new B point matching that projection
            dot_clipped = np.clip(dot, 0, magB[None, :])
            pB[oob_A] = b0_bcast[oob_A] + dot_clipped[oob_A, None] * _B_bcast[oob_A]
        if np.any(oob_B):
            # Get the projection of the new B point onto the A vectors
            dot = np.einsum("ijk,ijk->ij", pB - a0_bcast, _A_bcast)
            # Get a new A point matching that projection
            dot_clipped = np.clip(dot, 0, magA[:, None])
            pA[oob_B] = a0_bcast[oob_B] + dot_clipped[oob_B, None] * _A_bcast[oob_B]
    else:
        # If there is no clamping, we use the scale value as-is to get the
        # closest points between the rays
        pA = a0_exp + t0[:, :, None] * _A_exp
        pB = b0_exp + t1[:, :, None] * _B_exp

    # Handle parallel case
    parallel = denom == 0
    if np.any(parallel):
        # Get the (N, N) projection of each point in b0 onto each unit vector
        # in A. Subtract the (N, 1) projection of each point in a0 along their
        # unit vector. By subtracting them, we calculating whether b0 falls
        # "ahead" or "behind" of a0, along the relevant axis (_A)
        d0 = np.einsum("ij,kj->ik", _A, b0) - np.einsum("ij,ij->i", _A, a0).reshape(
            -1, 1
        )
        if clamp:
            # Same logic as d0, but with a1 and b1
            d1 = np.einsum("ij,kj->ik", _A, b1) - np.einsum("ij,ij->i", _A, a0).reshape(
                -1, 1
            )

            # Check in which of the (N, N) combinations b0 and b1 fall before
            # a0, forming an (N, N) boolean mask
            before = (d0 <= 0) & (d1 <= 0) & parallel
            # Same logic for after
            after = (d0 >= magA[:, None]) & (d1 >= magA[:, None]) & parallel
            # If an A and B vector is parallel and partially overlapping,
            # middle will be true (N, N) boolean mask
            middle = parallel & ~(before | after)

            if np.any(before):
                # Set pA to a0 where relevant
                pA[before] = a0[:, None, :][before]
                # Set pB to whichever endpoint is closer to a0
                pB[before] = np.where(
                    np.abs(d0[before])[:, None] < np.abs(d1[before])[:, None],
                    b0[None, :, :][before],
                    b1[None, :, :][before],
                )

            if np.any(after):
                # Set pA to a1 where relevant
                pA[after] = a1[:, None, :][after]
                # Set pB to whichever endpoint is closer to a1
                pB[after] = np.where(
                    np.abs(d0[after])[:, None] < np.abs(d1[after])[:, None],
                    b0[None, :, :][after],
                    b1[None, :, :][after],
                )

            if np.any(middle):
                # Clip d0 along the A vectors (from 0 to the magnitude of A)
                t_mid = np.clip(
                    d0[middle], 0, np.broadcast_to(magA[:, None], d0.shape)[middle]
                )

                # Get the broadcast components that align with the state we
                # care about (middle)
                a0_bcast = np.broadcast_to(a0[:, None, :], pA.shape)
                _A_bcast = np.broadcast_to(_A[:, None, :], pA.shape)
                a0_mid = a0_bcast[middle]
                _A_mid = _A_bcast[middle]

                # Reproject onto the A vectors
                pA[middle] = a0_mid + t_mid[:, None] * _A_mid

                # Get the vector from the new pA points to b0
                b0_bcast = np.broadcast_to(b0[None, :, :], pB.shape)
                a2b = b0_bcast[middle] - pA[middle]

                # Then subtract the component that is along the A vectors, and
                # that gives us the final pB location
                alongA = np.einsum("ij,ij->i", a2b, _A_mid)[:, None] * _A_mid
                pB[middle] = pA[middle] + (a2b - alongA)

        else:
            # If we're not clamping, arbitrarily set the parallel "closest
            # points" to b0 and the matching point on A
            a0_bcast = np.broadcast_to(a0[:, None, :], pA.shape)
            b0_bcast = np.broadcast_to(b0[None, :, :], pB.shape)
            _A_bcast = np.broadcast_to(_A[:, None, :], pA.shape)
            pA[parallel] = (
                a0_bcast[parallel] + d0[:, :, None][parallel] * _A_bcast[parallel]
            )
            pB[parallel] = b0_bcast[parallel]

    # pA and pB are existing (N, N, 3) matrices. In addition, calculate the
    # (N, N) distances between each pair
    return pA, pB, np.linalg.norm(pA - pB, axis=2)


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
    closest_points = []
    pA, pB, _ = compute_approximate_ray_intersections(
        a0=starts, a1=ends, b0=starts, b1=ends, clamp=True
    )
    mask = ~np.eye(starts.shape[0], dtype=bool)
    return np.mean(np.vstack([pA[mask], pB[mask]]), axis=0)
