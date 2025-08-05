import json
from pathlib import Path
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
np.set_printoptions(suppress=True)

from geograypher.cameras import MetashapeCameraSet


Z = np.array([0, 0, 1])


image_dir = Path("/home/eric/Pictures/test_lines/")
image_paths = sorted(image_dir.glob("*JPG"))
camera_file = Path("/home/eric/Pictures/test_lines/000936_01_cameras.xml")
dtm_file = Path("/home/eric/Pictures/test_lines/000936_01_dtm-ptcloud.tif")
original_image_folder = Path("/data/03_input-images/000936/000936-01/00/")

# Load camera set
cameras = MetashapeCameraSet(
    camera_file=camera_file,
    image_folder=image_dir,
    original_image_folder=original_image_folder,
    validate_images=True,
)

for cam in cameras:
    print(cam.get_image_filename())

def generate_pixel_grid_ij(height: int, width: int, step: int) -> np.ndarray:
    """Returns (N, 2) np.ndarray of pixel coordinates in (i, j) order"""
    i_vals = np.arange(0, height, step)
    j_vals = np.arange(0, width, step)
    ii, jj = np.meshgrid(i_vals, j_vals, indexing="ij")
    return np.stack([ii.ravel(), jj.ravel()], axis=1)


def plot_pixel_arrows(start_pts: np.ndarray, end_pts: np.ndarray, height: int, width: int, image_path: str = None):
    """
    Plot black dots at start_pts and lines from start_pts to end_pts.
    If image_path is provided, the plot is overlaid on that image.
    Otherwise, it's overlaid on a white background.

    Parameters:
    - start_pts: (N, 2) array of (i, j) pixel coordinates (rows, cols)
    - end_pts: (N, 2) array of (i, j) pixel coordinates
    - image_path: Optional path to an image file
    """
    if image_path:
        image = np.array(Image.open(image_path))
    else:
        image = np.ones((height, width, 3), dtype=np.uint8) * 255

    figure, axis = plt.subplots()
    axis.imshow(image)

    # Draw black dots at start points
    axis.scatter(start_pts[:, 1], start_pts[:, 0], c="black", s=10)

    # Draw arrows from start to end points
    for (i0, j0), (i1, j1) in zip(start_pts, end_pts):
        axis.arrow(j0, i0, j1 - j0, i1 - i0, color="red", head_width=1.5, head_length=2, length_includes_head=True)

    axis.set_xlim(0, width)
    axis.set_ylim(height, 0)  # Flip y-axis to image coordinates
    axis.set_aspect("equal")
    axis.axis("off")
    plt.tight_layout()
    plt.show()

vis_grid = generate_pixel_grid_ij(cam.image_height, cam.image_width, step=200)
vis_cast = cam.cast_rays(vis_grid)[1::2]
vis_perspective = cam.decast_rays(vis_cast + Z)
plot_pixel_arrows(start_pts=vis_grid, end_pts=vis_perspective, height=cam.image_height, width=cam.image_width, image_path=cam.get_image_filename())

# import ipdb; ipdb.set_trace()

cache = {}
for cam in tqdm(cameras):
    save_grid = generate_pixel_grid_ij(cam.image_height, cam.image_width, step=20)
    save_cast = cam.cast_rays(save_grid)[1::2]
    save_perspective = cam.decast_rays(save_cast + Z)
    cache[str(cam.get_image_filename())] = {
        "src": save_grid.tolist(),
        "dest": save_perspective.tolist(),
    }
cache_path = image_dir / "grid20.json"
json.dump(cache, cache_path.open("w"))
