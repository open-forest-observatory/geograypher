from multiview_prediction_toolkit.segmentation.derived_segmentors import BrightnessSegmentor 
from multiview_prediction_toolkit.segmentation.segmentor import SegmentorPhotogrammetryCameraSet
from multiview_prediction_toolkit.cameras import MetashapeCameraSet
from multiview_prediction_toolkit.config import DEFAULT_CAM_FILE, DEFAULT_IMAGES_FOLDER
import numpy as np
import matplotlib.pyplot as plt

segmentor = BrightnessSegmentor()

base_camera_set = MetashapeCameraSet(DEFAULT_CAM_FILE, DEFAULT_IMAGES_FOLDER)
segmentation_camera_set = SegmentorPhotogrammetryCameraSet(base_camera_set=base_camera_set, segmentor=segmentor)
inds = np.arange(0, base_camera_set.n_cameras(), 100)

for i in inds:
    img = segmentation_camera_set.get_raw_image_by_index(i)
    segmented = segmentation_camera_set.get_image_by_index(i)
    f, ax = plt.subplots(1,3)
    ax[0].imshow(img)
    ax[1].imshow(segmented[...,0])
    ax[2].imshow(segmented[...,1])
    plt.show()


