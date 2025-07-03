import numpy as np
import pytest

from geograypher.utils.numeric import (
    compute_approximate_ray_intersection,
    triangulate_rays_lstsq,
)


def normalize(vector):
    return vector / np.linalg.norm(vector)


class TestComputeApproximateRayIntersection:

    @pytest.mark.parametrize(
        "a0, a1, b0, b1, clamp, exp_pA, exp_pB, exp_dist",
        (
            # Test simple intersection
            (
                [0, 0, 0],
                [2, 0, 0],
                [1, 1, 0],
                [1, -1, 0],
                False,
                [1, 0, 0],
                [1, 0, 0],
                0,
            ),
            # Test simple miss case
            (
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 1],
                [0, 0, 1],
                False,
                [0, 0, 0],
                [0, 0, 1],
                1,
            ),
            # One much bigger than the other, no clamp
            (
                [-10, 0, 0],
                [10, 0, 0],
                [0, 1, 0],
                [1, 2, 0],
                False,
                [-1, 0, 0],
                [-1, 0, 0],
                0,
            ),
            # One much bigger than the other, clamp
            (
                [-10, 0, 0],
                [10, 0, 0],
                [0, 1, 0],
                [1, 2, 0],
                True,
                [0, 0, 0],
                [0, 1, 0],
                1,
            ),
            # Somewhat aligned, clamp
            (
                [1, 0, 0],
                [2, -4, 0],
                [-1, 0, 0],
                [-2, 8, 0],
                True,
                [1, 0, 0],
                [-1, 0, 0],
                2,
            ),
            # Parallel, with overlap
            (
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [1, 1, 0],
                True,
                None,
                None,
                1,
            ),
            # Parallel, with no overlap
            (
                [0, 0, 0],
                [1, 0, 0],
                [4, 4, 0],
                [5, 4, 0],
                True,
                [1, 0, 0],
                [4, 4, 0],
                5,
            ),
        ),
    )
    def test_basic(self, a0, a1, b0, b1, clamp, exp_pA, exp_pB, exp_dist):
        """Test a variety of ray intersections."""

        # These tests should work regardless of orientation
        for A0, A1 in [(a0, a1), (a1, a0)]:
            for B0, B1 in [(b0, b1), (b1, b0)]:
                pA, pB, dist = compute_approximate_ray_intersection(
                    np.array(A0),
                    np.array(A1),
                    np.array(B0),
                    np.array(B1),
                    clamp=clamp,
                )
                if exp_pA is None:
                    assert pA is None
                else:
                    assert np.allclose(pA, exp_pA)
                if exp_pB is None:
                    assert pB is None
                else:
                    assert np.allclose(pB, exp_pB)
                assert np.isclose(dist, exp_dist)


# class TestTriangulateRaysLstsq:
#     def test_basic(self):
#         # Three rays intersecting at (1, 2, 3)
#         starts = np.array([
#             [1, 2, 0],
#             [1, 0, 3],
#             [0, 2, 3],
#         ])
#         directions = np.array([
#             [0, 0, 1],
#             [0, 1, 0],
#             [1, 0, 0],
#         ])
#         point = triangulate_rays_lstsq(starts, directions)
#         assert np.allclose(point, [1, 2, 3])

#     def test_sampling(self):
#         """
#         Check a high number of randomized discrete line segments that do not cross.
#         The least squares solution should never extend past the endpoints

#         o        o
#          \       |
#           \      |
#            o-----o  valid/desired
#             .    .
#              .   .
#               .  .
#                . .
#                 -   invalid/undesired
#         """

#         # Note that the main direction of travel is +-Y
#         p0_0 = np.array([-1, 1, 0])
#         p0_1 = np.array([-1, -1, 0])
#         p1_0 = np.array([1, 1, 0])
#         p1_1 = np.array([1, -1, 0])

#         # Calculate N attempts of (4, 3) noise
#         all_noise = np.random.normal(scale=0.05, size=(1000, 4, 3))
#         for noise in all_noise:
#             A0 = p0_0 + noise[0]
#             A1 = p0_1 + noise[1]
#             a = normalize(A1 - A0)
#             B0 = p1_0 + noise[2]
#             B1 = p1_1 + noise[3]
#             b = normalize(B1 - B0)
#             point0 = triangulate_rays_lstsq(
#                 starts=np.vstack([A0, B0]),
#                 directions=np.vstack([a, b]),
#             )
#             point1 = triangulate_rays_lstsq(
#                 starts=np.vstack([A1, B1]),
#                 directions=np.vstack([-a, -b]),
#             )
#             point2 = triangulate_rays_lstsq(
#                 starts=np.vstack([A0, B1]),
#                 directions=np.vstack([a, -b]),
#             )

#             text = f"A: ({A0}, {A1}), B: ({B0}, {B1})"

#             # Test bidirectionality
#             assert np.allclose(point0, point1), f"{text} mismatched for {(point0, point1)}"
#             assert np.allclose(point0, point2), f"{text} mismatched for {(point0, point2)}"

#             # Test capping (vector direction was mostly along Y)
#             assert point0[1] <= np.max([A0[1], B0[1]]) + 1e-4, f"{text} out of bounds for {point0[1]}"
#             assert point0[1] >= np.max([A1[1], B1[1]]) - 1e-4, f"{text} out of bounds for {point0[1]}"
