import pyproj
import pytest
import numpy as np

from geograypher.meshes import TexturedPhotogrammetryMesh
from geograypher.utils.example_data import (
    create_non_overlapping_points,
    create_scene_mesh,
)
from geograypher.constants import EARTH_CENTERED_EARTH_FIXED_CRS


@pytest.fixture
def basic_scene(n_boxes=4, n_cylinders=5, n_cones=3, random_seed=42):
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

    textured_mesh = TexturedPhotogrammetryMesh(mesh=mesh_geometry, input_CRS=26910)

    return (textured_mesh, labels_gdf)


@pytest.mark.parametrize(
    "crs,inplace",
    [
        (4236, True),
        (4236, False),
        (4978, True),
        (4978, False),
        ("EPSG:32610", True),
        ("EPSG:32610", False),
    ],
)
def test_reprojection(
    basic_scene: TexturedPhotogrammetryMesh,
    crs: pyproj.CRS,
    inplace: bool,
):
    textured_mesh, labels_gdf = basic_scene
    textured_mesh.reproject_CRS(target_CRS=crs, inplace=inplace)


def test_internal_coordinates(basic_scene):
    assert basic_scene[0].CRS == EARTH_CENTERED_EARTH_FIXED_CRS


# def test_roundtrip(basic_scene):
#    textured_mesh = basic_scene[0]
#
#    vertices_initial = np.array(textured_mesh.pyvista_mesh.verts)
#    textured_mesh.reproject_CRS(4236, inplace=True)
#
#    vertices_4236 = np.array(textured_mesh.pyvista_mesh.verts)
#    assert not np.allclose(vertices_initial, vertices_4236)
#
#    textured_mesh.reproject_CRS(26910, inplace=True)
#
#    textured_mesh.reproject_CRS
