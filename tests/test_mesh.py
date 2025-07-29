import pytest
import pyproj

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
