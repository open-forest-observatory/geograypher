import numpy as np
import pyvista as pv
from geopandas import GeoDataFrame
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
from shapely import Polygon


def create_non_overlapping_points(
    n_points, distance_thresh=1, size=10, random_seed=None
):
    np.random.seed(random_seed)
    all_points = (np.random.rand(1, 2) - 0.5) * size
    while all_points.shape[0] < n_points:
        new_point = (np.random.rand(1, 2) - 0.5) * size

        dist_from_existing = cdist(new_point, all_points)
        if np.min(dist_from_existing) > distance_thresh:
            all_points = np.concatenate((all_points, new_point), axis=0)

    return all_points


def extract_polygon(mesh: pv.PolyData):
    xy_points = mesh.points[:, :2]
    hull = ConvexHull(xy_points)
    return Polygon(xy_points[hull.vertices])


def create_scene_mesh(
    box_centers=(),
    cylinder_centers=(),
    cone_centers=(),
    cylinder_radius=0.5,
    cone_radius=0.5,
    box_size=1 / np.sqrt(2.0),
    grid_size=(20, 20),
    add_ground=True,
    ground_resolution=200,
):
    box_meshes = []
    box_polygons = []
    ID = 0.0
    for x, y in box_centers:
        x_min = x - box_size / 2.0
        x_max = x + box_size / 2.0
        y_min = y - box_size / 2.0
        y_max = y + box_size / 2.0

        box = pv.Box((x_min, x_max, y_min, y_max, 0, box_size), quads=False)
        box["ID"] = np.full(box.n_cells, fill_value=ID)
        box_meshes.append(box)
        box_polygons.append(extract_polygon(box))
        ID += 1.0

    cylinder_meshes = []
    cylinder_polygons = []
    for x, y in cylinder_centers:
        cylinder = pv.Cylinder(
            (x, y, 0.5), direction=(0, 0, 1), radius=cylinder_radius, resolution=10
        ).triangulate()
        cylinder["ID"] = np.full(cylinder.n_cells, fill_value=ID)
        cylinder_meshes.append(cylinder)
        cylinder_polygons.append(extract_polygon(cylinder))
        ID += 1.0

    cone_meshes = []
    cone_polygons = []
    for x, y in cone_centers:
        cone = pv.Cone(
            (x, y, 0.5),
            direction=(0, 0, -1),
            radius=cone_radius,
            resolution=12,
        ).triangulate()
        cone["ID"] = np.full(cone.n_cells, fill_value=ID)
        cone_meshes.append(cone)
        cone_polygons.append(extract_polygon(cone))
        ID += 1.0

    merged_mesh = pv.merge(box_meshes + cylinder_meshes + cone_meshes)
    labels_gdf = GeoDataFrame(
        {
            "name": ["cube"] * len(box_meshes)
            + ["cylinder"] * len(cylinder_meshes)
            + ["cone"] * len(cone_meshes)
        },
        geometry=box_polygons + cylinder_polygons + cone_polygons,
    )

    if add_ground:
        # Add the ground plane
        object_points = merged_mesh.points[np.isclose(merged_mesh.points[:, 2], 0)]
        grid_size_x, grid_size_y = grid_size
        # Add the corner points of the ground plane
        grid = np.meshgrid(
            np.linspace(-grid_size_x / 2, grid_size_x / 2, num=ground_resolution),
            np.linspace(-grid_size_y / 2, grid_size_y / 2, num=ground_resolution),
        )
        grid = [x.flatten() for x in grid]
        ground_points = np.vstack(grid + [np.zeros_like(grid[0])]).T

        ground_points = pv.PolyData(ground_points)
        object_points = pv.PolyData(object_points)
        combined = object_points + ground_points
        # Triangulate between all the vertices
        ground_surf = (combined).delaunay_2d()
        # Set the ID to nan
        ground_surf["ID"] = np.full(ground_surf.n_cells, fill_value=np.nan)
        # Merge the ground plane with the other actors
        merged_mesh = merged_mesh + ground_surf
    return merged_mesh, labels_gdf
