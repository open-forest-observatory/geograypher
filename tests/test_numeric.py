import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from geograypher.utils.numeric import (
    compute_approximate_ray_intersections,
    intersection_average,
)


class TestComputeApproximateRayIntersection:

    # These are all formatted as
    # [a0[0], a0[1], ...] - (N, 3) - all the starting points of the A vectors
    # [a1[0], a1[1], ...] - (N, 3) - all the ending points of the A vectors
    # [b0[0], b0[1], ...] - (N, 3) - all the starting points of the B vectors
    # [b1[0], b1[1], ...] - (N, 3) - all the ending points of the B vectors
    # bool - whether to clamp or calculate for infinite rays
    # [A[0] to B[0], A[1] to B[0], ...] - (N, N, 3) - expected "closest point" on the A vectors
    # [A[0] to B[1], A[1] to B[1], ...]
    # [...]
    # [A[0] to B[0], A[1] to B[0], ...] - (N, N, 3) - expected "closest point" on the B vectors
    # [A[0] to B[1], A[1] to B[1], ...]
    # [...]
    # [A[0] to B[0], A[1] to B[0], ...] - (N, N) - expected distance between closest points
    # [A[0] to B[1], A[1] to B[1], ...]
    # [...]
    # bool - whether the lines are parallel (ignore closest point logic)
    @pytest.mark.parametrize(
        "a0, a1, b0, b1, clamp, exp_pA, exp_pB, exp_dist, parallel",
        (
            # Test simple intersections
            (
                [[0, 0, 0], [0, 0, 0]],
                [[2, 0, 0], [2, 2, 0]],
                [[1, 1, 0], [2, 0, 0]],
                [[1, -1, 0], [1, 1, 0]],
                False,
                [[[1, 0, 0], [2, 0, 0]], [[1, 1, 0], [1, 1, 0]]],
                [[[1, 0, 0], [2, 0, 0]], [[1, 1, 0], [1, 1, 0]]],
                [[0, 0], [0, 0]],
                False,
            ),
            # Test simple miss cases, infinite rays (no clamping)
            (
                [[0, 0, 0], [0, 1, -1]],
                [[1, 0, 0], [1, 1, -1]],
                [[0, 1, 1], [0, -4, 2]],
                [[0, 0, 1], [2, -2, 2]],
                False,
                [[[0, 0, 0], [4, 0, 0]], [[0, 1, -1], [5, 1, -1]]],
                [[[0, 0, 1], [4, 0, 2]], [[0, 1, 1], [5, 1, 2]]],
                [[1, 2], [2, 3]],
                False,
            ),
            # One much bigger than the other, no clamp
            (
                [[-10, 0, 0]],
                [[10, 0, 0]],
                [[0, 1, 0]],
                [[1, 2, 0]],
                False,
                [[[-1, 0, 0]]],
                [[[-1, 0, 0]]],
                [[0]],
                False,
            ),
            # One much bigger than the other, clamp
            (
                [[-10, 0, 0], [-10, 0, 1]],
                [[10, 0, 0], [10, 0, 1]],
                [[0, 1, 0], [0, 1, 2]],
                [[1, 2, 0], [0, 1, 1]],
                True,
                [[[0, 0, 0], [0, 0, 0]], [[0, 0, 1], [0, 0, 1]]],
                [[[0, 1, 0], [0, 1, 1]], [[0, 1, 0], [0, 1, 1]]],
                [[1, np.sqrt(2)], [np.sqrt(2), 1]],
                False,
            ),
            # Somewhat aligned, clamp
            (
                [[1, 0, 0], [0, 0, 0]],
                [[2, -4, 0], [1, 10, 0]],
                [[-1, 0, 0], [3, -5, 0]],
                [[-2, 8, 0], [3, -10, 0]],
                True,
                [[[1, 0, 0], [2, -4, 0]], [[0, 0, 0], [0, 0, 0]]],
                [[[-1, 0, 0], [3, -5, 0]], [[-1, 0, 0], [3, -5, 0]]],
                [[2, np.sqrt(2)], [1, np.sqrt(34)]],
                False,
            ),
            # Parallel, with overlap
            (
                [[0, 0, 0], [0, -1, 0]],
                [[1, 0, 0], [1, -1, 0]],
                [[0, 1, 0], [0.4, 2, 0], [-1, 3, 0], [0.9, 4, 0]],
                [[1, 1, 0], [0.6, 2, 0], [0.1, 3, 0], [2, 4, 0]],
                True,
                None,
                None,
                [[1, 2, 3, 4], [2, 3, 4, 5]],
                True,
            ),
            # Parallel, with no overlap
            (
                [[0, 0, 0]],
                [[1, 0, 0]],
                [[4, 4, 0]],
                [[5, 4, 0]],
                True,
                None,
                None,
                [[5]],
                True,
            ),
            # Parallel, no clamp
            (
                [[0, 0, 0], [0, -1, 0]],
                [[1, 0, 0], [1, -1, 0]],
                [[0, 1, 0], [0.4, 2, 0], [-1, 3, 0], [0.9, 4, 0]],
                [[1, 1, 0], [0.6, 2, 0], [0.1, 3, 0], [2, 4, 0]],
                False,
                None,
                None,
                [[1, 2, 3, 4], [2, 3, 4, 5]],
                True,
            ),
        ),
    )
    def test_basic(self, a0, a1, b0, b1, clamp, exp_pA, exp_pB, exp_dist, parallel):
        """Test a variety of ray intersections."""

        # These tests should work regardless of orientation
        for A in [(a0, a1), (a1, a0)]:
            for B in [(b0, b1), (b1, b0)]:
                # We should also be able to swap A and B
                for (c0, c1), (d0, d1), exp_pC, exp_pD, mirror in [
                    (A, B, exp_pA, exp_pB, False),
                    (B, A, exp_pB, exp_pA, True),
                ]:
                    pA, pB, dist = compute_approximate_ray_intersections(
                        a0=np.array(c0),
                        a1=np.array(c1),
                        b0=np.array(d0),
                        b1=np.array(d1),
                        clamp=clamp,
                    )

                    # If the lines are parallel the "closest point" gets murky,
                    # just test the distance
                    if not parallel:
                        if exp_pC is None:
                            assert pA is None
                        else:
                            if mirror:
                                exp_pC = np.swapaxes(exp_pC, 0, 1)
                            assert np.allclose(pA, exp_pC)
                        if exp_pD is None:
                            assert pB is None
                        else:
                            if mirror:
                                exp_pD = np.swapaxes(exp_pD, 0, 1)
                            assert np.allclose(pB, exp_pD)

                    if mirror:
                        test_dist = np.array(exp_dist).T
                    else:
                        test_dist = exp_dist
                    assert np.allclose(dist, test_dist)


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
