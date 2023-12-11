from pathlib import Path

from multiview_mapping_toolkit.cameras import MetashapeCameraSet
from multiview_mapping_toolkit.config import (DATA_FOLDER, DEFAULT_CAM_FILE,
                                                 DEFAULT_IMAGES_FOLDER,
                                                 DEFAULT_LOCAL_MESH)
from multiview_mapping_toolkit.meshes import \
    HeightAboveGroundPhotogrammertryMesh

IMAGES_FOLDER = Path(DATA_FOLDER, "training", "images")
RENDERS_FOLDER = Path(DATA_FOLDER, "training", "labels")
LABELS_FOLDER = Path(DATA_FOLDER, "composite_georef", "segmented")


IMAGES_FOLDER.mkdir(exist_ok=True, parents=True)
RENDERS_FOLDER.mkdir(exist_ok=True)

camera_set = MetashapeCameraSet(
    camera_file=DEFAULT_CAM_FILE, image_folder=DEFAULT_IMAGES_FOLDER
)
# Create mesh texture
mesh = HeightAboveGroundPhotogrammertryMesh(
    DEFAULT_LOCAL_MESH, downsample_target=0.25, ground_height_threshold=None
)
# mesh.vis()
mesh.save_renders_pytorch3d(camera_set=camera_set, render_folder="heightmaps")
