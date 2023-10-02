import numpy as np
import typing


class Segmentor:
    def __init__(self):
        pass

    def setup(self, **kwargs) -> None:
        """This is for things like loading a model. It's fine to not override it if there's no setup"""
        pass

    def segment_image(self, image: np.ndarray, **kwargs):
        """Produce a segmentation mask for an image

        Args:
            image (np.ndarray): np
        """
        raise NotImplementedError("Abstract base class")

    def segment_image_batch(self, images: typing.List[np.ndarray], **kwargs):
        """
        Segment a batch of images, to potentially use full compute capacity. The current implementation
        should be overriden when there is a way to get improvements

        Args:
            images (typing.List[np.ndarray]): The list of images
        """
        segmentations = []

        for image in images:
            segmentation = self.segment_image(image, **kwargs)
            segmentations.append(segmentation)
        return segmentations
