import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from geograypher.utils.numeric import (
    calc_communities,
    calc_graph_weights,
    chunk_slices,
    compute_approximate_ray_intersections,
    format_graph_edges,
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


def test_chunk_slices():
    expected = [
        (slice(0, 200, None), slice(0, 200, None), True),
        (slice(0, 200, None), slice(200, 400, None), False),
        (slice(0, 200, None), slice(400, 500, None), False),
        (slice(200, 400, None), slice(200, 400, None), True),
        (slice(200, 400, None), slice(400, 500, None), False),
        (slice(400, 500, None), slice(400, 500, None), True),
    ]
    for expect, actual in zip(expected, chunk_slices(N=500, step=200)):
        assert expect == actual


@pytest.mark.parametrize("distance", [3, 5, 10])
@pytest.mark.parametrize(
    "alter_kwargs,expected_indices",
    [
        # Start off with the default kwargs
        ({}, [[6 + 3, 6 + 4]]),
        # Go off diagonal, getting more indices
        (
            {"islice": slice(0, 6, None)},
            [
                [1, 6 + 1],
                [3, 6 + 4],
                [5, 6 + 4],
            ],
        ),
        # Go off diagonal on the other side of diagonal, should be no matches
        ({"jslice": slice(0, 6, None)}, []),
        # Make some ray indices match, there should be no matches where the
        # ray indices match
        (
            {
                "islice": slice(0, 6, None),
                "ray_IDs": np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 3, 11]),
            },
            [
                [1, 6 + 1],
                [5, 6 + 4],
            ],
        ),
    ],
)
def test_format_graph_edges(distance, alter_kwargs, expected_indices):

    def check_types(edges):
        assert isinstance(edges, list)
        for edge in edges:
            assert len(edge) == 3
            assert isinstance(edge, tuple)
            assert isinstance(edge[0], int)
            assert isinstance(edge[1], int)
            assert isinstance(edge[2], dict)
            assert len(edge[2]) == 1
            assert "weight" in edge[2]

    # With no modifications this is a diagonal set of chunks, most
    # indices should be pruned
    kwargs = {
        "i_inds": np.array([1, 3, 5]),
        "j_inds": np.array([1, 4, 4]),
        "islice": slice(6, 12, None),
        "jslice": slice(6, 12, None),
        "dist": np.ones((6, 6)) * distance,
        "ray_IDs": np.array(range(12)),
    }
    for key, value in alter_kwargs.items():
        kwargs[key] = value

    edges = format_graph_edges(**kwargs)
    check_types(edges)

    assert np.allclose(
        np.array([[a, b, c["weight"]] for a, b, c in edges]),
        [row + [1 / distance] for row in expected_indices],
    )


@pytest.fixture
def line_kwargs():
    """6 segments in 3D, some intersecting, some not."""
    ray_starts = np.array(
        [
            [0, 0, 0],  # Group A
            [0, 1, 0],  # Group A
            [2, 0, 0],  # Group B
            [2, 0.75, 0],  # Group B
            [2, 0, 0.5],  # Group B
            [10, 0, 0],  # Far enough away it should be entirely excluded
        ]
    )
    segment_ends = np.array(
        [
            [0, 1, 0],
            [0, 1, 1],
            [2, 1, 0],
            [2, 0.75, 1],
            [2, 0, 2],
            [20, 0, 0],
        ]
    )
    return {
        "starts": ray_starts,
        "ends": segment_ends,
        "ray_IDs": np.array(range(len(ray_starts))),
    }


class TestCalcGraphWeights:

    @pytest.mark.parametrize(
        "transform",
        [
            lambda x: x,  # No transform
            lambda x: x**2,  # Squared
            lambda x: x**3,  # Cubed
        ],
    )
    @pytest.mark.parametrize("to_file", [True, False])
    def test_calc_graph_weights(self, tmp_path, line_kwargs, to_file, transform):

        output = calc_graph_weights(
            similarity_threshold=1.5,
            out_dir=tmp_path if to_file else None,
            min_dist=1e-4,
            step=2,
            transform=transform,
            **line_kwargs,
        )

        # Load the results if necessary
        if to_file:
            assert isinstance(output, Path)
            assert output.suffix == ".json"
            assert output.is_file()
            edges = json.load(output.open("r"))
        else:
            edges = output

        # Should be a list of (i, j, dict)
        assert isinstance(edges, list)
        assert len(edges) > 0
        for edge in edges:
            assert len(edge) == 3
            i, j, d = edge
            for index in (i, j):
                assert isinstance(index, int)
                assert 0 <= index < 6
            assert isinstance(d, dict)
            assert len(d) == 1
            assert "weight" in d

        # Based on how the line segments were constructed, we know the
        # weights should be as follows
        expected = {
            (0, 1): 1 / transform(1e-4),
            (2, 3): 1 / transform(1e-4),
            (2, 4): 1 / transform(0.5),
            (3, 4): 1 / transform(0.75),
        }
        edge_dict = {(i, j): d["weight"] for i, j, d in edges}
        assert set(edge_dict.keys()) == set(expected.keys())
        for indices in edge_dict.keys():
            assert np.isclose(
                edge_dict[indices], expected[indices]
            ), f"{indices} mismatched"


@pytest.fixture
def sample_graph_inputs():
    """Sample inputs for a simple graph, has several obvious communities."""
    return {
        "starts": np.array(
            [
                # Group 1
                [0, 0, 0],
                [1, 1, 0],
                [-1, 0, 0],
                # Group 2
                [3.1, 0, 10],
                [3.1, 0, 10],
                [3.05, 0.1, 10],
                # Group 3
                [4, 4, 0],
                [4.5, 4, 0],
            ]
        ),
        "ends": np.array(
            [
                # Group 1
                [0, 0, 2],
                [-1, -1, 2],
                [1, 0, 2],
                # Group 2
                [3, 0.05, 5],
                [3.05, 0.05, 5],
                [3.05, 0.1, 5],
                # Group 3
                [6, 6, 0],
                [4, 6.5, 0],
            ]
        ),
        # Weights are arbitrary, just need to be positive
        "positive_edges": [
            (0, 1, {"weight": 1.0}),
            (0, 2, {"weight": 1.0}),
            (3, 4, {"weight": 1.0}),
            (3, 5, {"weight": 1.0}),
            (6, 7, {"weight": 1.0}),
        ],
    }


class TestCalcCommunities:

    @pytest.mark.parametrize("to_file", [True, False])
    @pytest.mark.parametrize("give_transform", [True, False])
    def test_basic(self, tmp_path, to_file, give_transform, sample_graph_inputs):

        # If requested, give a filler transform, just to see if it gets used
        output = calc_communities(
            **sample_graph_inputs,
            out_dir=tmp_path if to_file else None,
            transform_to_epsg_4978=np.eye(4) if give_transform else None,
        )

        if to_file:
            assert isinstance(output, Path)
            assert output.suffix == ".npz"
            assert output.is_file()
            result = dict(np.load(output))
        else:
            result = output

        assert isinstance(result, dict)
        assert "ray_IDs" in result
        assert "community_points" in result

        # If we gave a transform, it should result in lat/lon points.
        # Don't check them for correctness
        assert ("community_points_latlon" in result) == give_transform

        # Check that the ray IDs break down into the expected communities
        assert result["ray_IDs"].shape == (8,)
        assert np.all(result["ray_IDs"] == np.array([0, 0, 0, 1, 1, 1, 2, 2]))

        # Check that the community points are approximately correct
        assert result["community_points"].shape == (3, 3)
        assert np.allclose(
            result["community_points"],
            np.array([[0, 0, 1], [3, 0, 7.5], [5, 5, 0]]),
            atol=1,
        )

    @pytest.mark.parametrize("give_transform", [True, False])
    def test_empty(self, give_transform, sample_graph_inputs):
        """Test that when no graph edges are given we get empty arrays back."""

        result = calc_communities(
            starts=np.array([]),
            ends=np.array([]),
            positive_edges=[],
            transform_to_epsg_4978=np.eye(4) if give_transform else None,
        )

        assert isinstance(result, dict)
        assert "ray_IDs" in result
        assert "community_points" in result

        assert result["ray_IDs"].shape == (0,)
        assert result["community_points"].shape == (0, 3)

        if give_transform:
            assert "community_points_latlon" in result
            assert result["community_points_latlon"].shape == (0, 3)
