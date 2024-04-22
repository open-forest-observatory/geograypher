import typing

import numpy as np


class Segmentor:
    def __init__(self, num_classes=None):
        self.num_classes = num_classes

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

    @staticmethod
    def inds_to_one_hot(
        inds_image: np.ndarray,
        num_classes: typing.Union[int, None] = None,
        ignore_ind: int = 255,
    ) -> np.ndarray:
        """Convert an image of indices to a one-hot, one-per-channel encoding

        Args:
            inds_image (np.ndarray): Image of integer indices. (m, n)
            num_classes (int, None): The number of classes. If None, computed as the max index provided. Default None
            ignore_ind (inte, optional): This index is an ignored class

        Returns:
            np.ndarray: (m, n, num_classes) boolean array with one channel filled with a True, all else False
        """
        if num_classes is None:
            inds_image_copy = inds_image.copy()
            # Mask out ignore ind so it's not used in computation
            inds_image_copy[inds_image_copy == ignore_ind] == 0
            num_classes = np.max(inds_image_copy) + 1

        one_hot_array = np.zeros(
            (inds_image.shape[0], inds_image.shape[1], num_classes), dtype=bool
        )
        # Iterate up to max ind, not num_classes to avoid wasted computation when there won't be matches
        for i in range(num_classes):
            # TODO determine if there are any more efficient ways to do this
            # Maybe create all these slices and then concatenate
            # Or test equality with an array that has all the values in it
            one_hot_array[..., i] = inds_image == i

        return one_hot_array
