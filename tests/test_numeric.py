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
        "a0, a1, b0, b1, clamp, exp_pA, exp_pB, exp_dist, parallel",
        (
            # Test simple intersections
            (
                [0, 0, 0],
                [2, 0, 0],
                [[1, 1, 0], [2, 0, 0]],
                [[1, -1, 0], [1, 1, 0]],
                False,
                [[1, 0, 0], [2, 0, 0]],
                [[1, 0, 0], [2, 0, 0]],
                [0, 0],
                False,
            ),
            # Test simple miss cases, infinite rays (no clamping)
            (
                [0, 0, 0],
                [1, 0, 0],
                [[0, 1, 1], [0, -4, 2]],
                [[0, 0, 1], [2, -2, 2]],
                False,
                [[0, 0, 0], [4, 0, 0]],
                [[0, 0, 1], [4, 0, 2]],
                [1, 2],
                False,
            ),
            # One much bigger than the other, no clamp
            (
                [-10, 0, 0],
                [10, 0, 0],
                [[0, 1, 0]],
                [[1, 2, 0]],
                False,
                [[-1, 0, 0]],
                [[-1, 0, 0]],
                [0],
                False,
            ),
            # One much bigger than the other, clamp
            (
                [-10, 0, 0],
                [10, 0, 0],
                [[0, 1, 0], [0, 1, 2]],
                [[1, 2, 0], [0, 1, 1]],
                True,
                [[0, 0, 0], [0, 0, 0]],
                [[0, 1, 0], [0, 1, 1]],
                [1, np.sqrt(2)],
                False,
            ),
            # Somewhat aligned, clamp
            (
                [1, 0, 0],
                [2, -4, 0],
                [[-1, 0, 0], [3, -5, 0]],
                [[-2, 8, 0], [3, -10, 0]],
                True,
                [[1, 0, 0], [2, -4, 0]],
                [[-1, 0, 0], [3, -5, 0]],
                [2, np.sqrt(2)],
                False,
            ),
            # Parallel, with overlap
            (
                [0, 0, 0],
                [1, 0, 0],
                [[0, 1, 0], [0.4, 2, 0], [-1, 3, 0], [0.9, 4, 0]],
                [[1, 1, 0], [0.6, 2, 0], [0.1, 3, 0], [2, 4, 0]],
                True,
                None,
                None,
                [1, 2, 3, 4],
                True,
            ),
            # Parallel, with no overlap
            (
                [0, 0, 0],
                [1, 0, 0],
                [[4, 4, 0]],
                [[5, 4, 0]],
                True,
                None,
                None,
                [5],
                True,
            ),
        ),
    )
    def test_basic(self, a0, a1, b0, b1, clamp, exp_pA, exp_pB, exp_dist, parallel):
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

                # If the lines are parallel the "closest point" gets murky,
                # just test the distance
                if not parallel:
                    if exp_pA is None:
                        assert pA is None
                    else:
                        assert np.allclose(pA, exp_pA)
                    if exp_pB is None:
                        assert pB is None
                    else:
                        assert np.allclose(pB, exp_pB)

                assert np.allclose(dist, exp_dist)


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
        # Parallel behavior is to snap to a0 for lack of something better to do
        assert np.allclose(intersection_average(starts, ends), [0, 0.5, 0])
