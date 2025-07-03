import numpy as np
import pytest

from geograypher.utils.numeric import (
    compute_approximate_ray_intersection,
    intersection_average,
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


class TestIntersectionAverage:
    def test_intersections(self):
        """Two segments crossing at (0,0,0)."""
        starts = np.array(
            [
                [-1, 0, 0],
                [0, -1, 0],
            ]
        )
        ends = np.array(
            [
                [1, 0, 0],
                [0, 1, 0],
            ]
        )
        # Both segments cross at (0,0,0), so average should be (0,0,0)
        assert np.allclose(intersection_average(starts, ends), [0, 0, 0])

    def test_complex(self):
        """Make a more complex, non-intersecting test."""
        starts = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 2, 0],
            ]
        )
        ends = np.array(
            [
                [0, 0, -2],
                [2, 0, -2],
                [0, 1, -2],
            ]
        )
        # [0] and [1] closest is [0, 0, 0] and [1, 0, 0]
        # [0] and [2] closest is [0, 0, -2] and [0, 1, -2]
        # [1] and [2] closest is [1 1/3, 0, -2/3] and [0, 1 1/3, -1 1/3]
        assert np.allclose(intersection_average(starts, ends), [7 / 18, 7 / 18, -1])

    def test_parallel(self):
        """Two parallel segments, no intersection."""
        starts = np.array(
            [
                [0, 0, 0],
                [0, 1, 0],
            ]
        )
        ends = np.array(
            [
                [1, 0, 0],
                [1, 1, 0],
            ]
        )
        # Should fall back to average of all endpoints
        all_points = np.concatenate([starts, ends], axis=0)
        expected = np.mean(all_points, axis=0)
        assert np.allclose(intersection_average(starts, ends), expected)
