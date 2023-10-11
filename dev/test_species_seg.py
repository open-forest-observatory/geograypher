from sqlite3 import DatabaseError
from multiview_prediction_toolkit.cameras.derived_cameras import MetashapeCameraSet
from multiview_prediction_toolkit.meshes import (
    GeodataPhotogrammetryMesh,
    TexturedPhotogrammetryMesh,
)
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

from multiview_prediction_toolkit.segmentation.derived_segmentors import LookUpSegmentor
from multiview_prediction_toolkit.segmentation.segmentor import (
    SegmentorPhotogrammetryCameraSet,
)

IMAGE_SCALE = 0.25
IMAGES_FOLDER = Path(DATA_FOLDER, "training", "images")
RENDERS_FOLDER = Path(DATA_FOLDER, "training", "labels")
LABELS_FOLDER = Path(DATA_FOLDER, "composite_georef", "segmented")

CREATE_LABELS = False

IMAGES_FOLDER.mkdir(exist_ok=True, parents=True)
RENDERS_FOLDER.mkdir(exist_ok=True)

camera_set = MetashapeCameraSet(
    camera_file=DEFAULT_CAM_FILE, image_folder=DEFAULT_IMAGES_FOLDER
)
if CREATE_LABELS:
    # Create mesh texture
    geodata_mesh = GeodataPhotogrammetryMesh(
        DEFAULT_LOCAL_MESH,
        geo_point_file=DEFAULT_GEO_POINTS_FILE,
    )
    geodata_mesh.vis(cmap="tab10")
    breakpoint()

    for i in tqdm(range(camera_set.n_cameras())):
        image = camera_set.get_image_by_index(i, image_scale=IMAGE_SCALE)
        image_path = camera_set.get_camera_by_index(i).image_filename
        label_mask = geodata_mesh.render_pytorch3d(
            camera_set, image_scale=IMAGE_SCALE, camera_index=i
        )
        np.save(Path(RENDERS_FOLDER, f"{i:06d}.npy"), label_mask)
        os.symlink(image_path, Path(IMAGES_FOLDER, f"{i:06d}{Path(image_path).suffix}"))
else:
    mesh = TexturedPhotogrammetryMesh(DEFAULT_LOCAL_MESH, downsample_target=0.25)
    segmentor = LookUpSegmentor(
        base_folder=DEFAULT_IMAGES_FOLDER, lookup_folder=LABELS_FOLDER
    )

    segmentor_camera_set = SegmentorPhotogrammetryCameraSet(
        camera_set, segmentor=segmentor
    )
    img = segmentor_camera_set.get_image_by_index(0, image_scale=IMAGE_SCALE)

    species, _, _ = mesh.aggregate_viewpoints_pytorch3d(
        segmentor_camera_set, image_scale=IMAGE_SCALE
    )
    max_class = np.argmax(species, axis=1)
    mesh.vis(vis_scalars=max_class, cmap="tab10", interactive=True)
    breakpoint()
