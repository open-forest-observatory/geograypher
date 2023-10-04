import numpy as np
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
