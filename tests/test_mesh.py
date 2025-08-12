import numpy as np
import pyproj
import pytest
import pyvista as pv

from geograypher.constants import EARTH_CENTERED_EARTH_FIXED_CRS
from geograypher.meshes import TexturedPhotogrammetryMesh
from geograypher.utils.example_data import (
    create_non_overlapping_points,
    create_scene_mesh,
)


# Because this is a pytest.fixture, when mesh_and_labels is an argument to a subsequent test, the
# value will be that of this function's return
@pytest.fixture
def mesh_and_labels(n_boxes=4, n_cylinders=5, n_cones=3, random_seed=42):
    """
    Create a random map with several classes of objects on a flat plane. Retuns the mesh geometry
    and a 2D dataframe of the object locations.
    """
    # Create object centers
    points = create_non_overlapping_points(
        n_points=(n_boxes + n_cylinders + n_cones), random_seed=random_seed
    )
    # Create the geometry and labels dataframe
    mesh_geometry, labels_gdf = create_scene_mesh(
        box_centers=points[:n_boxes],
        cylinder_centers=points[n_boxes : (n_boxes + n_cylinders)],
        cone_centers=points[
            (n_boxes + n_cylinders) : (n_boxes + n_cylinders + n_cones)
        ],
    )

    return (mesh_geometry, labels_gdf)


class TestMeshProjection:
    def test_internal_coordinates(self, mesh_and_labels, input_CRS=32610):
        # Create a mesh in an arbitrary CRS
        textured_photogrammetry_mesh = TexturedPhotogrammetryMesh(
            mesh_and_labels[0], input_CRS=input_CRS
        )
        # Ensure that the coordinates are internally represented in the earth-centered, earth-fixed
        # frame
        assert textured_photogrammetry_mesh.CRS == EARTH_CENTERED_EARTH_FIXED_CRS

    def test_crop_valid(self, mesh_and_labels, input_CRS=32610):
        mesh_geometry, labels = mesh_and_labels

        # Define the CRS to interpret the labels in
        labels.set_crs(input_CRS, inplace=True)

        # Create a full mesh with no cropping
        full_mesh = TexturedPhotogrammetryMesh(mesh_geometry, input_CRS=input_CRS)
        # Create a mesh cropped to the bounds of the labels
        cropped_mesh = TexturedPhotogrammetryMesh(
            mesh_geometry, input_CRS=input_CRS, ROI=labels
        )

        # Check that there are more points in the full mesh than the cropped one
        assert full_mesh.pyvista_mesh.n_points > cropped_mesh.pyvista_mesh.n_points
        # And the cropped mesh has a nonzero number of points
        assert cropped_mesh.pyvista_mesh.n_points > 0

    @pytest.mark.parametrize("input_CRS", [32610])
    # Three output CRS, the initial one, lat-long, and another cartesian one
    @pytest.mark.parametrize("output_CRS", [32610, 4326, 3311])
    def test_vertex_reprojection(
        self, mesh_and_labels, input_CRS: pyproj.CRS, output_CRS: pyproj.CRS
    ):
        """Test that the vertices represented as a geodataframe are in the correct location

        Args:
            mesh_and_labels: The mesh and geospatial labels from the test fixture.
            input_CRS (pyproj.CRS): CRS to initially represent the data in. Should be cartesian.
            output_CRS (pyproj.CRS): CRS to transform to.
        """
        mesh, labels = mesh_and_labels

        # Create the labels and textured mesh in the same CRS
        labels.set_crs(input_CRS, inplace=True)
        textured_photogrammetry_mesh = TexturedPhotogrammetryMesh(
            mesh, input_CRS=input_CRS
        )

        # Get the mesh vertices in the new CRS
        vertices_geospatial = textured_photogrammetry_mesh.get_verts_geodataframe(
            output_CRS
        )
        # Reproject the labels to the new CRs
        labels.to_crs(output_CRS, inplace=True)

        labels_bounds = labels.dissolve().geometry[0]

        vertices_within_labels = vertices_geospatial.within(labels_bounds)

        # Some vertices should not be within bounds
        assert vertices_within_labels.sum() < len(vertices_geospatial)
        # But at least one needs to be
        assert vertices_within_labels.sum() > 0


class TestExportCoveringMeshes:

    @pytest.fixture
    def sample_mesh(self):
        np.random.seed(33)

        # Create a simple mesh for testing, with a range from (0-1) in
        # all three dimensions. XY should form a grid, Z will be random.
        xy = np.stack(
            np.meshgrid(np.linspace(0, 1, 10), np.linspace(0, 1, 10)), axis=-1
        ).reshape(-1, 2)
        points = np.hstack([xy, np.random.random((xy.shape[0], 1))])
        mesh = pv.PolyData(points).delaunay_2d()

        assert isinstance(mesh, pv.PolyData)
        textured_mesh = TexturedPhotogrammetryMesh(
            mesh, input_CRS=EARTH_CENTERED_EARTH_FIXED_CRS
        )
        return textured_mesh

    def test_basic_covering_mesh_creation(self, sample_mesh):
        """Test basic functionality of creating covering meshes"""
        N = 3
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
            N=N, z_buffer=z_buffer
        )

        # Get original meshes without buffer for comparison
        upper_mesh_no_buffer, lower_mesh_no_buffer = sample_mesh.export_covering_meshes(
            N=N, z_buffer=(0, 0)
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
        mesh = TexturedPhotogrammetryMesh(
            pv.PolyData(),
            input_CRS=EARTH_CENTERED_EARTH_FIXED_CRS,
        )
        with pytest.raises(AssertionError):
            mesh.export_covering_meshes(N=4, z_buffer=(1.0,))

        # No mesh provided
        mesh.pyvista_mesh = None
        with pytest.raises(AssertionError):
            mesh.export_covering_meshes(N=4)

    @pytest.mark.parametrize("subsample", [3, 5, 9])
    def test_subsample_behavior(self, sample_mesh, subsample):
        """Test subsampling behavior"""
        N = 10

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
        mesh = TexturedPhotogrammetryMesh(
            pv.PolyData(),
            input_CRS=EARTH_CENTERED_EARTH_FIXED_CRS,
        )
        upper_mesh, lower_mesh = mesh.export_covering_meshes(N=4)

        # Should return empty meshes
        assert upper_mesh.n_points == 0
        assert lower_mesh.n_points == 0
