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

segmentor = BrightnessSegmentor()
mesh = TexturedPhotogrammetryMesh(
    mesh_filename=DEFAULT_LOCAL_MESH, downsample_target=DOWNSAMPLE_TARGET
)

base_camera_set = MetashapeCameraSet(DEFAULT_CAM_FILE, DEFAULT_IMAGES_FOLDER)
segmentation_camera_set = SegmentorPhotogrammetryCameraSet(
    base_camera_set=base_camera_set, segmentor=segmentor
)
inds = np.arange(0, base_camera_set.n_cameras(), 10)

normalized_face_colors, face_colors, counts = mesh.aggregate_viewpoints_pytorch3d(
    segmentation_camera_set, image_scale=IMAGE_SCALE, camera_inds=inds
)
mesh.show_face_textures(normalized_face_colors)
mesh.show_face_textures(normalized_face_colors[:, 0])
mesh.show_face_textures(counts)

for i in inds:
    img = segmentation_camera_set.get_raw_image_by_index(i, image_scale=IMAGE_SCALE)
    segmented = segmentation_camera_set.get_image_by_index(i, image_scale=IMAGE_SCALE)
    f, ax = plt.subplots(1, 3)
    ax[0].imshow(img)
    ax[1].imshow(segmented[..., 0])
    ax[2].imshow(segmented[..., 1])
    plt.show()
