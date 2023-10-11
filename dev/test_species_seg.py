from multiview_prediction_toolkit.cameras.derived_cameras import MetashapeCameraSet
from multiview_prediction_toolkit.meshes import GeodataPhotogrammetryMesh
from multiview_prediction_toolkit.config import (
    DATA_FOLDER,
    DEFAULT_CAM_FILE,
    DEFAULT_IMAGES_FOLDER,
    DEFAULT_LOCAL_MESH,
    DEFAULT_GEO_POINTS_FILE,
)
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from imageio import imwrite
from tqdm import tqdm
import os

IMAGE_SCALE = 1
IMAGES_FOLDER = Path(DATA_FOLDER, "training", "images")
RENDERS_FOLDER = Path(DATA_FOLDER, "training", "labels")
IMAGES_FOLDER.mkdir(exist_ok=True, parents=True)
RENDERS_FOLDER.mkdir(exist_ok=True)


camera_set = MetashapeCameraSet(
    camera_file=DEFAULT_CAM_FILE, image_folder=DEFAULT_IMAGES_FOLDER
).get_subset_near_geofile(DEFAULT_GEO_POINTS_FILE)
mesh = GeodataPhotogrammetryMesh(
    DEFAULT_LOCAL_MESH, geo_point_file=DEFAULT_GEO_POINTS_FILE
)
mesh.vis(interactive=True, cmap="tab20")
for i in tqdm(range(camera_set.n_cameras())):
    image = camera_set.get_image_by_index(i, image_scale=IMAGE_SCALE)
    image_path = camera_set.get_camera_by_index(i).image_filename
    label_mask = mesh.render_pytorch3d(
        camera_set, image_scale=IMAGE_SCALE, camera_index=i
    )
    np.save(Path(RENDERS_FOLDER, f"{i:06d}.npy"), label_mask)
    os.symlink(image_path, Path(IMAGES_FOLDER, f"{i:06d}{Path(image_path).suffix}"))
