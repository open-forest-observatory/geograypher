import numpy as np
import pytest
import pyvista as pv

from geograypher.meshes import TexturedPhotogrammetryMesh


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
