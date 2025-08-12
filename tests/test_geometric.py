import numpy as np
import pytest
import pyvista as pv

from geograypher.utils.geometric import clip_line_segments


class TestClipLineSegments:

    @pytest.fixture
    def sample_boundaries(self):
        """
        Create two simple planar surfaces for testing, a lower plane at z=0
        and an upper plane at z=1
        """
        plane0 = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [1, 1, 0],
                [0, 1, 0],
            ]
        )
        plane1 = plane0.copy()
        plane1[:, 2] = 1

        return (
            pv.PolyData(plane1).delaunay_2d(),
            pv.PolyData(plane0).delaunay_2d(),
        )

    def test_basic_clipping(self, sample_boundaries):
        """Test rays that should intersect both surfaces"""

        # Above and below both surfaces
        origins = np.array([[0.5, 0.5, 2.0], [0.1, 0.3, -1.0]])
        # Downward and upward directions
        directions = np.array([[0, 0, -1], [0, 0, 1]])
        image_indices = list(range(len(origins)))

        starts, ends, dirs, indices = clip_line_segments(
            sample_boundaries, origins, directions, image_indices
        )

        # Check types and lengths
        for output in [starts, ends, dirs, indices]:
            assert isinstance(output, np.ndarray)
            assert len(output) == 2

        # Check some values by hand
        assert np.allclose(starts, [[0.5, 0.5, 1.0], [0.1, 0.3, 1.0]])
        assert np.allclose(ends, [[0.5, 0.5, 0.0], [0.1, 0.3, 0.0]])
        assert np.allclose(dirs, [[0, 0, -1], [0, 0, -1]])
        assert np.allclose(indices, image_indices)

    def test_new_direction(self, sample_boundaries):
        """
        Test that the final direction respects the boundary[0] → boundary[1]
        direction that lines get clipped in.
        """

        # Starts off travelling up
        origins = np.array([[0.5, 0.5, -2.0]])
        directions = np.array([[0, 0, 1]])
        image_indices = list(range(len(origins)))

        starts, ends, dirs, indices = clip_line_segments(
            sample_boundaries, origins, directions, image_indices
        )

        # Ends up travelling from boundary[0] to boundary[1]
        assert np.allclose(starts, [[0.5, 0.5, 1.0]])
        assert np.allclose(ends, [[0.5, 0.5, 0.0]])
        assert np.allclose(dirs, [[0, 0, -1]])
        assert np.allclose(indices, image_indices)

    def test_ray_limit(self, sample_boundaries):
        """Test that limiting the length of the rays correctly filters rays."""

        # Since ray length is measured between original starting point and the
        # lower boundary, make downward-facing rays that start different distances
        # away
        origins = np.array([[0.5, 0.5, 2.0], [0.7, 0.7, 1.1]])
        directions = np.array([[0, 0, -1], [0, 0, -1]])
        image_indices = [0] * len(origins)

        # Test with ray limit that should exclude the longer ray
        starts, ends, dirs, indices = clip_line_segments(
            sample_boundaries, origins, directions, image_indices, ray_limit=1.5
        )
        assert np.allclose(starts, [[0.7, 0.7, 1.0]])

        # Test with ray limit that should include the longer ray
        starts, ends, dirs, indices = clip_line_segments(
            sample_boundaries, origins, directions, image_indices, ray_limit=3.0
        )
        assert np.allclose(starts, [[0.5, 0.5, 1.0], [0.7, 0.7, 1.0]])

    def test_no_intersections(self, sample_boundaries):
        """Test rays that miss both surfaces."""
        origins = np.array(
            [
                [0.2, 0.2, 0.5],
                [0.3, 0.3, 0.5],
                [0.4, 0.4, 1.5],
                [0.5, 0.5, -1.5],
                [1.6, 1.6, 1.5],
            ]
        )
        directions = np.array(
            [
                [0, 0, -1],  # Will only intersect one boundary
                [1, 0, 0],  # Parallel to the boundaries
                [0, 0, -1],  # Should intersect
                [0, 0, 1],  # Should intersect
                [0, 0, -1],  # Should miss the boundaries based on start point
            ]
        )
        image_indices = list(range(len(origins)))

        starts, ends, dirs, indices = clip_line_segments(
            sample_boundaries, origins, directions, image_indices
        )

        assert np.allclose(starts, [[0.4, 0.4, 1.0], [0.5, 0.5, 1.0]])
        assert np.allclose(indices, [2, 3])

    def test_empty_inputs(self, sample_boundaries):
        starts, ends, dirs, indices = clip_line_segments(
            boundaries=sample_boundaries,
            origins=np.zeros((0, 3)),
            directions=np.zeros((0, 3)),
            image_indices=[],
        )
        assert len(starts) == 0
        assert len(ends) == 0
        assert len(dirs) == 0
        assert len(indices) == 0

    @pytest.mark.parametrize(
        "modify_kwargs,key_str",
        (
            [{"boundaries": [None]}, "2 boundaries required"],
            [{"boundaries": [None, None]}, "pv.PolyData required"],
            [
                {"origins": np.array([[0.5, 0.5, 2.0]])},
                "origins and directions mismatched",
            ],
            [
                {"directions": np.array([[0.5, 0.5, 2.0]])},
                "origins and directions mismatched",
            ],
            [
                {
                    "origins": np.array([[1, 1]]),
                    "directions": np.array([[1, 1]]),
                },
                "(N, 3) input arrays required",
            ],
            [{"image_indices": [0]}, "image indices mismatched"],
        ),
    )
    def test_errors(self, sample_boundaries, modify_kwargs, key_str):
        """Try feeding in a variety of invalid inputs."""

        # Valid arguments
        kwargs = {
            "boundaries": sample_boundaries,
            "origins": np.array([[0.5, 0.5, 2.0], [0.5, 0.5, 3.0]]),
            "directions": np.array([[0, 0, -1], [0, 0, -1]]),
            "image_indices": [0, 1],
        }
        # Replace the normal kwargs with bad substitutions
        for k, v in modify_kwargs.items():
            kwargs[k] = v

        with pytest.raises(ValueError) as ve:
            clip_line_segments(**kwargs)
        assert key_str in str(ve.value)
