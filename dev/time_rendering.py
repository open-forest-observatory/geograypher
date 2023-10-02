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

IMAGE_SCALES = np.geomspace(0.05, 1, num=20)
SAVEFILE = Path(VIS_FOLDER, "image_scale_timing.png")

camera_set = MetashapeCameraSet(
    camera_file=DEFAULT_CAM_FILE, image_folder=DEFAULT_IMAGES_FOLDER
)
CAMERA_INDICES = list(range(0, camera_set.n_cameras(), 100))

mesh = ColorPhotogrammetryMesh(mesh_filename=DEFAULT_LOCAL_MESH)

times_taken = []


for image_scale in IMAGE_SCALES:
    start_time = time.time()
    for ind in CAMERA_INDICES:
        mesh.render_pytorch3d(
            camera_set=camera_set, image_scale=image_scale, camera_index=ind
        )
    times_taken.append(float(time.time() - start_time))
    plt.xlabel("image scale")
    plt.ylabel(f"Time (s) taken to render {len(CAMERA_INDICES)} images")
    plt.plot(IMAGE_SCALES[: len(times_taken)], times_taken)
    plt.savefig(SAVEFILE)
    plt.clf()
