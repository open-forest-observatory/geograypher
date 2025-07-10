import json
import typing
from itertools import product

import numpy as np
from tqdm import tqdm


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
    a0: np.ndarray, a1: np.ndarray, b0: np.ndarray, b1: np.ndarray, clamp: bool = False
):
    """
    Compute closest points and distances between N line segments a0->a1 and b0->b1.
    Returns (N, N, 3), (N, N, 3), (N, N)
    """

    A = a1 - a0  # (N, 3)
    B = b1 - b0  # (N, 3)
    magA = np.linalg.norm(A, axis=1)  # (N,)
    magB = np.linalg.norm(B, axis=1)  # (N,)
    _A = A / magA[:, None]  # (N, 3)
    _B = B / magB[:, None]  # (N, 3)

    a0_exp = a0[:, None, :]  # (N, 1, 3)
    b0_exp = b0[None, :, :]  # (1, N, 3)
    _A_exp = _A[:, None, :]  # (N, 1, 3)
    _B_exp = _B[None, :, :]  # (1, N, 3)

    cross = np.cross(_A_exp, _B_exp)  # (N, N, 3)
    denom = np.linalg.norm(cross, axis=2) ** 2  # (N, N)

    t = b0_exp - a0_exp  # (N, N, 3)

    detA = np.einsum("ijk,ijk->ij", np.cross(t, _B_exp), cross)  # (N, N)
    detB = np.einsum("ijk,ijk->ij", np.cross(t, _A_exp), cross)  # (N, N)

    denom_safe = np.where(denom == 0, 1, denom)
    t0 = detA / denom_safe
    t1 = detB / denom_safe

    if clamp:
        t0_clamped = np.clip(t0, 0, magA[:, None])  # (N, N)
        t1_clamped = np.clip(t1, 0, magB[None, :])  # (N, N)

        pA = a0_exp + t0_clamped[:, :, None] * _A_exp
        pB = b0_exp + t1_clamped[:, :, None] * _B_exp

        oob_A = (t0 < 0) | (t0 > magA[:, None])
        oob_B = (t1 < 0) | (t1 > magB[None, :])

        # Broadcast a0 and _A to (N, N, 3)
        a0_bcast = np.broadcast_to(a0[:, None, :], pA.shape)
        _A_bcast = np.broadcast_to(_A[:, None, :], pA.shape)
        b0_bcast = np.broadcast_to(b0[None, :, :], pB.shape)
        _B_bcast = np.broadcast_to(_B[None, :, :], pB.shape)

        if np.any(oob_A):
            dot = np.einsum("ijk,ijk->ij", pA - b0_bcast, _B_bcast)
            dot_clipped = np.clip(dot, 0, magB[None, :])
            pB[oob_A] = b0_bcast[oob_A] + dot_clipped[oob_A, None] * _B_bcast[oob_A]

        if np.any(oob_B):
            dot = np.einsum("ijk,ijk->ij", pB - a0_bcast, _A_bcast)
            dot_clipped = np.clip(dot, 0, magA[:, None])
            pA[oob_B] = a0_bcast[oob_B] + dot_clipped[oob_B, None] * _A_bcast[oob_B]
    else:
        pA = a0_exp + t0[:, :, None] * _A_exp
        pB = b0_exp + t1[:, :, None] * _B_exp

    # Handle parallel case
    parallel = denom == 0
    if np.any(parallel):
        d0 = np.einsum("ij,kj->ik", _A, b0) - np.einsum("ij,ij->i", _A, a0)[:, None]
        if clamp:
            d1 = np.einsum("ij,kj->ik", _A, b1) - np.einsum("ij,ij->i", _A, a0)[:, None]

            before = (d0 <= 0) & (d1 <= 0) & parallel
            after = (d0 >= magA[:, None]) & (d1 >= magA[:, None]) & parallel
            middle = parallel & ~(before | after)

            if np.any(before):
                pA[before] = a0[:, None, :][before]
                pB[before] = np.where(
                    np.abs(d0[before])[:, None] < np.abs(d1[before])[:, None],
                    b0[None, :, :][before],
                    b1[None, :, :][before],
                )

            if np.any(after):
                pA[after] = a1[:, None, :][after]
                pB[after] = np.where(
                    np.abs(d0[after])[:, None] < np.abs(d1[after])[:, None],
                    b0[None, :, :][after],
                    b1[None, :, :][after],
                )

            if np.any(middle):
                t_mid = np.clip(
                    d0[middle], 0, np.broadcast_to(magA[:, None], d0.shape)[middle]
                )

                a0_bcast = np.broadcast_to(a0[:, None, :], pA.shape)
                _A_bcast = np.broadcast_to(_A[:, None, :], pA.shape)

                a0_mid = a0_bcast[middle]
                _A_mid = _A_bcast[middle]

                pA[middle] = a0_mid + t_mid[:, None] * _A_mid

                b0_bcast = np.broadcast_to(b0[None, :, :], pB.shape)
                a2b = b0_bcast[middle] - pA[middle]

                alongA = np.einsum("ij,ij->i", a2b, _A_mid)[:, None] * _A_mid
                pB[middle] = pA[middle] + (a2b - alongA)

        else:
            pA[parallel] = (
                a0[:, None, :][parallel]
                + d0[:, :, None][parallel] * _A[:, None, :][parallel]
            )
            pB[parallel] = b0[None, :, :][parallel]

    return pA, pB, np.linalg.norm(pA - pB, axis=2)  # (N, N)


def SUMS(p, oob, v0_bcast, dot_clipped, _V_bcast):
    p[oob] = v0_bcast[oob] + dot_clipped[oob, None] * _V_bcast[oob]


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
    pA, pB, _ = compute_approximate_ray_intersection(
        a0=starts, a1=ends, b0=starts, b1=ends, clamp=True
    )
    mask = ~np.eye(starts.shape[0], dtype=bool)
    return np.mean(np.vstack([pA[mask], pB[mask]]), axis=0)


def calc_graph_weights(
    line_segments_file, similarity_threshold, out_dir, min_dist=1e-6, step=5000
):
    """
    Arguments:
        min_dist (float, optional): Limits the minimum intersection distance to some
            arbitrary small number to avoid div by 0
    """
    data = np.load(line_segments_file)
    starts = data["ray_starts"]
    ends = data["segment_ends"]
    ray_IDs = data["ray_IDs"]

    # Calculate the indices for ray-ray intersections
    positive_edges = []

    # Calculate and filter intersection distances
    # For memory reasons, we need to iterate over blocks. When the number of segments starts
    # getting very large, the matrices for calculating ray intersections take a great
    # deal of RAM
    num_steps = len(starts) // step + 1
    total = int(num_steps * (num_steps + 1) / 2)
    for islice, jslice, diagonal in tqdm(
        chunk_slices(N=len(starts), step=step),
        total=total,
        desc="Calculating graph weights",
    ):
        _, _, dist = compute_approximate_ray_intersection(
            a0=starts[islice],
            a1=ends[islice],
            b0=starts[jslice],
            b1=ends[jslice],
            clamp=True,
        )
        if diagonal:
            np.fill_diagonal(dist, np.nan)
        dist[dist > similarity_threshold] = np.nan
        dist[dist < min_dist] = min_dist

        # Determine which intersections are valid, represented by finite values
        i_inds, j_inds = np.where(np.isfinite(dist))
        positive_edges.extend(
            make_indices(i_inds, j_inds, islice, jslice, dist, ray_IDs)
        )

    path = out_dir / "positive_edges.json"
    with path.open("w") as file:
        json.dump(positive_edges, file)
    return path


def chunk_slices(
    N: int, step: int
) -> typing.Iterator[typing.Tuple[slice, slice, bool]]:
    """
    Yield slices for (step, step) chunks of an (N, N) square matrix.

    Each yielded value is (islice, jslice, is_diag), where:
    - islice: slice along the first axis
    - jslice: slice along the second axis
    - is_diag: True if the chunk is in the upper triangle (including diagonal)
    """
    ranges = range(0, N, step)
    for i, j in product(ranges, repeat=2):
        if j >= i:  # upper triangle including diagonal
            islice = slice(i, min(i + step, N))
            jslice = slice(j, min(j + step, N))
            yield islice, jslice, i == j


def make_indices(i_inds, j_inds, islice, jslice, dist, ray_IDs):
    return [
        (
            int(i) + islice.start,
            int(j) + jslice.start,
            {"weight": float(1 / dist[i, j])},
        )
        for i, j in zip(i_inds, j_inds)
        if (i + islice.start < j + jslice.start)
        and (ray_IDs[i + islice.start] != ray_IDs[j + jslice.start])
    ]
