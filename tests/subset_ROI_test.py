# %%
from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np
from scipy.spatial.transform import Rotation
from shapely import MultiPolygon, Point, Polygon
from tqdm import tqdm

from geograypher.cameras.cameras import PhotogrammetryCameraSet
from geograypher.constants import MATPLOTLIB_PALLETE, VIS_FOLDER
from geograypher.utils.example_data import (
    create_non_overlapping_points,
    create_scene_mesh,
)

# Where to save content
CONCEPT_FIGURE_ROOT = Path(VIS_FOLDER, "concept_figure_content")
IMAGE_FOLDER = Path(CONCEPT_FIGURE_ROOT, "realistic_images")
FIGURES_SAVE_FOLDER = Path(CONCEPT_FIGURE_ROOT, "figures")
LABEL_IMAGES_FOLDER = Path(CONCEPT_FIGURE_ROOT, "labeled_images")

# Save vis or show them in the notebook
SAVE_VIS = True

# Number of map elements
N_BOXES = 5
N_CYLINDERS = 5
N_CONES = 5
# Random seed for object locations
MAP_RANDOM_SEED = 42
# Discritization of the ground plane
GROUND_RESOLUTION = 200
# Scale of the frustum
VIS_FRUSTUM_SCALE = 1

# Mapping from integer IDs to human-readable labels
IDS_TO_LABELS = {0: "cone", 1: "cube", 2: "cylinder"}

# Range of hues for each object
HUE_RANGE_DICT = {"cone": 0.1, "cylinder": 0.2, "cube": 0.1}
# Only used for realist
REALISTIC_COLOR_DICT = {
    "cone": MATPLOTLIB_PALLETE[0],
    "cylinder": MATPLOTLIB_PALLETE[2],
    "cube": MATPLOTLIB_PALLETE[1],
}

# Camera set params
CAM_HEIGHT = 10
CAM_DIST_FROM_CENTER = 10
CAM_PITCH = 225

CAM_INTRINSICS = {
    0: {"f": 4000, "cx": 0, "cy": 0, "image_width": 3000, "image_height": 2200}
}

points = create_non_overlapping_points(
    n_points=(N_BOXES + N_CYLINDERS + N_CONES), random_seed=MAP_RANDOM_SEED
)

mesh, labels_gdf = create_scene_mesh(
    box_centers=points[:N_BOXES],
    cylinder_centers=points[N_BOXES : (N_BOXES + N_CYLINDERS)],
    cone_centers=points[(N_BOXES + N_CYLINDERS) : (N_BOXES + N_CYLINDERS + N_CONES)],
    add_ground=True,
    ground_resolution=GROUND_RESOLUTION,
)

def make_color_gradient(color, number, hue_range=None):
    if hue_range is None:
        hue_range = 1 / (number * 2)

    hsv_color = matplotlib.colors.rgb_to_hsv(color)
    hue_start = hsv_color[0] - hue_range / 2
    hue_end = hsv_color[0] + hue_range / 2
    hues = np.linspace(hue_start, hue_end, number) % 1.0
    shifted_HSVs = [np.concatenate(([hue], hsv_color[1:]), axis=0) for hue in hues]
    rgb_values = [
        matplotlib.colors.hsv_to_rgb(shifted_hue) for shifted_hue in shifted_HSVs
    ]
    rgb_values = np.vstack(rgb_values)

    rgb_values = rgb_values / 255.0

    return rgb_values


colors_per_face = np.full((mesh.n_cells, 3), fill_value=0.5)
IDs_per_face = np.full((mesh.n_cells, 1), fill_value=np.nan)

for i, (name, group) in enumerate(labels_gdf.groupby("name")):
    num = len(group)
    gradient = make_color_gradient(
        REALISTIC_COLOR_DICT[name], num, hue_range=HUE_RANGE_DICT[name]
    )
    # Indices into the original dataset
    IDs = group.index.to_numpy()
    # TODO rename
    for ID, color in zip(IDs, gradient):
        matching = mesh["ID"] == ID
        colors_per_face[matching, :] = color
        IDs_per_face[matching, :] = i

t_vecs = (
    (0, 0, CAM_HEIGHT),
    (0, CAM_DIST_FROM_CENTER, CAM_HEIGHT),
    (CAM_DIST_FROM_CENTER, 0, CAM_HEIGHT),
    (-CAM_DIST_FROM_CENTER, 0, CAM_HEIGHT),
    (0, -CAM_DIST_FROM_CENTER, CAM_HEIGHT),
)
# Create rotations in roll, pitch, yaw convention
r_vecs = (
    (180, 180, 0),  # nadir
    (180, CAM_PITCH, 0),  # oblique
    (90, CAM_PITCH, 0),  # oblique
    (270, CAM_PITCH, 0),  # oblique
    (0, CAM_PITCH, 0),  # oblique
)

# Create 4x4 transforms
cam_to_world_transforms = []
for r_vec, t_vec in zip(r_vecs, t_vecs):
    r_mat = Rotation.from_euler("ZXY", r_vec, degrees=True).as_matrix()
    transform = np.eye(4)
    transform[:3, :3] = r_mat
    transform[:3, 3] = t_vec
    cam_to_world_transforms.append(transform)

IMAGE_FOLDER.mkdir(exist_ok=True, parents=True)
# Note that these files do not exist yet, but will be later created
image_filenames = [Path(IMAGE_FOLDER, f"img_{i:03d}.png") for i in range(5)]

camera_set = PhotogrammetryCameraSet(
    cam_to_world_transforms=cam_to_world_transforms,
    intrinsic_params_per_sensor_type=CAM_INTRINSICS,
    image_folder=IMAGE_FOLDER,
    image_filenames=image_filenames,
)

#create different ROIs to test the camera_set.get_subset_ROI() function
polygon1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
polygon2 = Polygon([(1, 1), (2, 1), (2, 2), (1, 2)])
multi_polygon = MultiPolygon([polygon1, polygon2])
polygon_gdf1 = gpd.GeoDataFrame(data = {'name': ['Polygon 1'], 'geometry': [polygon1]})
polygon_gdf2 =  gpd.GeoDataFrame(data = {'name': ['Polygon 1', 'Polygon 2'], 'geometry': [polygon1, polygon2]})
multi_polygon_gdf = gpd.GeoDataFrame(data = {'name': ['MultiPolygon 1'], 'geometry': [multi_polygon]})
# test is_geospatial flag determination
assert camera_set.get_subset_ROI(ROI=polygon1).get_camera_locations() == camera_set.get_subset_ROI(ROI=polygon1, is_geospatial=False).get_camera_locations(), "geospatial determination is wrong"
# polygon ROI should produce same result as geodataframe containing the same polygon
assert camera_set.get_subset_ROI(ROI=polygon1).get_camera_locations() == camera_set.get_subset_ROI(ROI=polygon_gdf1).get_camera_locations(), "polygon ROI should produce same result as geodataframe containing the same polygon"
# multiple polygon rows in a gdf should dissolve and produce same result as a multipolygon gdf
assert camera_set.get_subset_ROI(ROI=multi_polygon_gdf).get_camera_locations() == camera_set.get_subset_ROI(ROI=polygon_gdf2).get_camera_locations(), "multiple polygon rows in a gdf should dissolve and produce same result as a multipolygon gdf"




