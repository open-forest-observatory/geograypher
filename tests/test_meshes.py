import numpy as np
import pytest
import pyvista as pv

from geograypher.meshes import TexturedPhotogrammetryMesh
from geograypher.meshes.meshes import clip_line_segments


class TestTexturedPhotogrammetryMeshCovering:

    @pytest.fixture
    def sample_mesh(self):
        np.random.seed(33)
        # Create a simple mesh for testing, with a range from (0-1) in
        # all three dimensions.
        points = np.random.random((100, 3))
        mesh = pv.PolyData(points).delaunay_2d()
        assert isinstance(mesh, pv.PolyData)
        textured_mesh = TexturedPhotogrammetryMesh(mesh)
        return textured_mesh

    def test_basic_covering_mesh_creation(self, sample_mesh):
        """Test basic functionality of creating covering meshes"""
        N = 4
        upper_mesh, lower_mesh = sample_mesh.export_covering_meshes(N=N)

        # Both meshes should be PolyData
        assert isinstance(upper_mesh, pv.PolyData)
        assert isinstance(lower_mesh, pv.PolyData)

        # Check that upper mesh z values are >= lower mesh z values
        assert np.all(upper_mesh.points[:, 2] >= lower_mesh.points[:, 2])

        # Check that the number of points is as expected
        expected_points = N * N
        assert upper_mesh.n_points == expected_points
        assert lower_mesh.n_points == expected_points

        print(upper_mesh.bounds)

        # Check that the bounds are correct
        for covering, use_max in [(upper_mesh, True), (lower_mesh, False)]:
            # First check the x and y bounds
            assert np.allclose(
                sample_mesh.pyvista_mesh.bounds[:4],
                np.array(covering.bounds[:4]),
            )
            # Then check the z bounds, min vs. max depends on which covering mesh
            if use_max:
                index = 5
            else:
                index = 4
            assert np.isclose(
                sample_mesh.pyvista_mesh.bounds[index],
                covering.bounds[index],
            )

    @pytest.mark.parametrize("z_buffer", [(1.0, -1.0), (0.5, -0.5), (2.0, 0.0)])
    def test_z_buffer(self, sample_mesh, z_buffer):
        """Test that z-buffer is correctly applied"""
        N = 4
        upper_mesh, lower_mesh = sample_mesh.export_covering_meshes(
            N=N, z_buffer_m=z_buffer
        )

        # Get original meshes without buffer for comparison
        upper_mesh_no_buffer, lower_mesh_no_buffer = sample_mesh.export_covering_meshes(
            N=N, z_buffer_m=(0, 0)
        )

        # Check buffer was applied
        assert np.allclose(
            upper_mesh.points[:, 2],
            upper_mesh_no_buffer.points[:, 2] + z_buffer[0],
            atol=1e-6,
        )
        assert np.allclose(
            lower_mesh.points[:, 2],
            lower_mesh_no_buffer.points[:, 2] + z_buffer[1],
            atol=1e-6,
        )

    def test_invalid_inputs(self):
        """Test that invalid inputs raise appropriate errors"""

        # Wrong number of z buffer values
        mesh = TexturedPhotogrammetryMesh(pv.PolyData())
        with pytest.raises(AssertionError):
            mesh.export_covering_meshes(N=4, z_buffer_m=(1.0,))

        # No mesh provided
        mesh.pyvista_mesh = None
        with pytest.raises(AssertionError):
            mesh.export_covering_meshes(N=4)

    @pytest.mark.parametrize("subsample", [2, 3, 5])
    def test_subsample_behavior(self, sample_mesh, subsample):
        """Test subsampling behavior"""
        N = 5

        # Get meshes without subsampling
        full_p, full_n = sample_mesh.export_covering_meshes(N=N)

        # Get meshes with subsampling
        sub_p, sub_n = sample_mesh.export_covering_meshes(
            N=N,
            subsample=subsample,
        )

        # Results should be similar but not identical due to subsampling.
        # In this case subsampling will knock out certain regions of
        # the grid, reducing the number of valid points.
        assert sub_p.n_points < full_p.n_points
        assert sub_n.n_points < full_n.n_points

    def test_empty_mesh(self):
        """Test behavior with empty mesh"""
        mesh = TexturedPhotogrammetryMesh(pv.PolyData())
        upper_mesh, lower_mesh = mesh.export_covering_meshes(N=4)

        # Should return empty meshes
        assert upper_mesh.n_points == 0
        assert lower_mesh.n_points == 0


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
