# %%
from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np
from scipy.spatial.transform import Rotation
from shapely import MultiPolygon, Polygon

from geograypher.cameras.cameras import PhotogrammetryCameraSet
from geograypher.constants import VIS_FOLDER

# Camera set params
CAM_HEIGHT = 10
CAM_DIST_FROM_CENTER = 10
CAM_PITCH = 225

CAM_INTRINSICS = {
    0: {"f": 4000, "cx": 0, "cy": 0, "image_width": 3000, "image_height": 2200}
}

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

camera_set = PhotogrammetryCameraSet(
    cam_to_world_transforms=cam_to_world_transforms,
    intrinsic_params_per_sensor_type=CAM_INTRINSICS,
)

# create different ROIs to test the camera_set.get_subset_ROI() function
# TODO: Implement pytest or another testing framework for more comprehensive testing
polygon1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
polygon2 = Polygon([(1, 1), (2, 1), (2, 2), (1, 2)])
multi_polygon = MultiPolygon([polygon1, polygon2])
polygon_gdf1 = gpd.GeoDataFrame(data={"name": ["Polygon 1"], "geometry": [polygon1]})
polygon_gdf2 = gpd.GeoDataFrame(
    data={"name": ["Polygon 1", "Polygon 2"], "geometry": [polygon1, polygon2]}
)
multi_polygon_gdf = gpd.GeoDataFrame(
    data={"name": ["MultiPolygon 1"], "geometry": [multi_polygon]}
)
# test is_geospatial flag determination
assert (
    camera_set.get_subset_ROI(ROI=polygon1).get_camera_locations()
    == camera_set.get_subset_ROI(
        ROI=polygon1, is_geospatial=False
    ).get_camera_locations()
), "geospatial determination is wrong"
# polygon ROI should produce same result as geodataframe containing the same polygon
assert (
    camera_set.get_subset_ROI(ROI=polygon1).get_camera_locations()
    == camera_set.get_subset_ROI(ROI=polygon_gdf1).get_camera_locations()
), "polygon ROI should produce same result as geodataframe containing the same polygon"
# multiple polygon rows in a gdf should dissolve and produce same result as a multipolygon gdf
assert (
    camera_set.get_subset_ROI(ROI=multi_polygon_gdf).get_camera_locations()
    == camera_set.get_subset_ROI(ROI=polygon_gdf2).get_camera_locations()
), "multiple polygon rows in a gdf should dissolve and produce same result as a multipolygon gdf"
