from multiview_prediction_toolkit.segmentation.derived_segmentors import (
    BrightnessSegmentor,
)
from multiview_prediction_toolkit.segmentation.segmentor import (
    SegmentorPhotogrammetryCameraSet,
)
from multiview_prediction_toolkit.cameras import MetashapeCameraSet
from multiview_prediction_toolkit.meshes import TexturedPhotogrammetryMesh
from multiview_prediction_toolkit.config import (
    DEFAULT_CAM_FILE,
    DEFAULT_IMAGES_FOLDER,
    DEFAULT_LOCAL_MESH,
)
import numpy as np
import matplotlib.pyplot as plt

DOWNSAMPLE_TARGET = 1.0
IMAGE_SCALE = 0.25

# Create the segmentor "model"
segmentor = BrightnessSegmentor()
# Load the mesh
mesh = TexturedPhotogrammetryMesh(
    mesh_filename=DEFAULT_LOCAL_MESH, downsample_target=DOWNSAMPLE_TARGET
)
# Load the set of cameras
base_camera_set = MetashapeCameraSet(DEFAULT_CAM_FILE, DEFAULT_IMAGES_FOLDER)
# Wrap these cameras in a segmentation algorithm
segmentation_camera_set = SegmentorPhotogrammetryCameraSet(
    base_camera_set=base_camera_set, segmentor=segmentor
)

# Use every fifth camera for texturing
camera_inds = np.arange(0, base_camera_set.n_cameras(), 2)
# Actually perform the texturing process, which returns three quantities
normalized_face_colors, face_colors, counts = mesh.aggregate_viewpoints_pytorch3d(
    segmentation_camera_set, image_scale=IMAGE_SCALE, camera_inds=camera_inds
)
# Show the normalized face color (two color)
mesh.show_face_textures(
    normalized_face_colors, screenshot_file="vis/two_class_segmentation.png"
)
# Show the first channel of the normalized fact colors (scalar)
mesh.show_face_textures(
    normalized_face_colors[:, 0], screenshot_file="vis/one_class_segmentation.png"
)
# Show the number of times each location was observed (scalar)
mesh.show_face_textures(counts, screenshot_file="vis/view_counts.png")

# Visualize a few segementation results
for i in np.arange(0, base_camera_set.n_cameras(), 200):
    img = segmentation_camera_set.get_raw_image_by_index(i, image_scale=IMAGE_SCALE)
    segmented = segmentation_camera_set.get_image_by_index(i, image_scale=IMAGE_SCALE)
    f, ax = plt.subplots(1, 3)
    ax[0].imshow(img)
    ax[1].imshow(segmented[..., 0])
    ax[2].imshow(segmented[..., 1])
    ax[0].set_title("Image")
    ax[1].set_title("Segmentation class 1")
    ax[2].set_title("Segmentation class 2")
    plt.show()
