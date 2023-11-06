from pathlib import Path

import numpy as np
from imageio import imread
from skimage.transform import resize

from multiview_prediction_toolkit.config import PATH_TYPE
from multiview_prediction_toolkit.segmentation import Segmentor


class BrightnessSegmentor(Segmentor):
    def __init__(self, brightness_threshold: float = np.sqrt(0.75)):
        self.brightness_threshold = brightness_threshold
        self.num_classes = 2

    def segment_image(self, image: np.ndarray, **kwargs):
        image_brightness = np.linalg.norm(image, axis=-1)
        thresholded_image = image_brightness > self.brightness_threshold
        class_index_image = thresholded_image.astype(np.uint8)
        one_hot_image = self.inds_to_one_hot(class_index_image)
        return one_hot_image


class LookUpSegmentor(Segmentor):
    def __init__(self, base_folder, lookup_folder, num_classes=10):
        self.base_folder = Path(base_folder)
        self.lookup_folder = lookup_folder
        self.num_classes = num_classes

    def segment_image(self, image: np.ndarray, filename: PATH_TYPE, image_scale: float):
        relative_path = Path(filename).relative_to(self.base_folder)
        lookup_path = Path(self.lookup_folder, relative_path)
        lookup_path = lookup_path.with_suffix(".png")

        image = imread(lookup_path)
        resized_image = resize(
            image,
            (int(image.shape[0] * image_scale), int(image.shape[1] * image_scale)),
            order=0,  # Nearest neighbor interpolation
        )
        one_hot_image = self.inds_to_one_hot(
            resized_image, num_classes=self.num_classes
        )
        return one_hot_image
