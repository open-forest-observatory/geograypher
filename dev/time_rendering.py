from multiview_prediction_toolkit.cameras import MetashapeCameraSet
from multiview_prediction_toolkit.config import (
    DEFAULT_CAM_FILE,
    DEFAULT_IMAGES_FOLDER,
    DEFAULT_LOCAL_MESH,
    VIS_FOLDER,
)
from multiview_prediction_toolkit.meshes import ColorPhotogrammetryMesh
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import time
from collections import defaultdict
import json

IMAGE_SCALES = np.geomspace(0.05, 1, num=20)
VIS_FILE = Path(VIS_FOLDER, "image_scale_timing.png")
SAVEFILE = Path(VIS_FOLDER, "image_scale_timing.json")

camera_set = MetashapeCameraSet(
    camera_file=DEFAULT_CAM_FILE, image_folder=DEFAULT_IMAGES_FOLDER
)
CAMERA_INDICES = list(range(0, camera_set.n_cameras(), 100))

mesh = ColorPhotogrammetryMesh(mesh_filename=DEFAULT_LOCAL_MESH)

times_taken = defaultdict(list)


def plot(timing_dict):
    x_axis = list(timing_dict.keys())
    values = np.array(timing_dict.values())
    means = np.mean(values, axis=1)
    stds = np.stds(values, axis=1)
    plt.xlabel("image scale")
    plt.ylabel(f"Time (s) taken to render {len(CAMERA_INDICES)} images")
    plt.plot(x_axis, means)
    plt.fill_between(x_axis, means - stds, means + stds, alpha=0.3)
    plt.savefig(VIS_FILE)
    plt.clf()


for image_scale in IMAGE_SCALES:
    for ind in CAMERA_INDICES:
        start_time = time.time()
        mesh.render_pytorch3d(
            camera_set=camera_set, image_scale=image_scale, camera_index=ind
        )
        times_taken[image_scale].append(float(time.time() - start_time))
    with open(SAVEFILE, "w") as fh:
        json.dump(dict(times_taken), fh)
