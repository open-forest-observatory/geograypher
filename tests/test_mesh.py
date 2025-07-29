import numpy as np
import pyproj
import pytest

from geograypher.constants import EARTH_CENTERED_EARTH_FIXED_CRS
from geograypher.meshes import TexturedPhotogrammetryMesh
from geograypher.utils.example_data import (
    create_non_overlapping_points,
    create_scene_mesh,
)


@pytest.fixture
def mesh_and_labels(n_boxes=4, n_cylinders=5, n_cones=3, random_seed=42):
    points = create_non_overlapping_points(
        n_points=(n_boxes + n_cylinders + n_cones), random_seed=random_seed
    )
    mesh_geometry, labels_gdf = create_scene_mesh(
        box_centers=points[:n_boxes],
        cylinder_centers=points[n_boxes : (n_boxes + n_cylinders)],
        cone_centers=points[
            (n_boxes + n_cylinders) : (n_boxes + n_cylinders + n_cones)
        ],
    )

    return (mesh_geometry, labels_gdf)


def test_internal_coordinates(mesh_and_labels, input_CRS=32610):
    # Create a mesh in an arbitrary CRS
    textured_photogrammetry_mesh = TexturedPhotogrammetryMesh(
        mesh_and_labels[0], input_CRS=input_CRS
    )
    # Ensure that the coordinates are internally represented in the earth-centered, earth-fixed
    # frame
    assert textured_photogrammetry_mesh.CRS == EARTH_CENTERED_EARTH_FIXED_CRS


def test_crop_valid(mesh_and_labels, input_CRS=32610):
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
