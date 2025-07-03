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
