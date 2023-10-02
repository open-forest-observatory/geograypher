from abc import abstractmethod
import numpy as np
import typing
from multiview_prediction_toolkit.cameras import (
    PhotogrammetryCameraSet,
    PhotogrammetryCamera,
)

from torch import NoneType


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
        inds_image: np.ndarray, num_classes: typing.Union[int, NoneType] = None
    ) -> np.ndarray:
        """Convert an image of indices to a one-hot, one-per-channel encoding

        Args:
            inds_image (np.ndarray): Image of integer indices. (m, n)
            num_classes (int, NoneType): The number of classes. If None, computed as the max index provided. Default None

        Returns:
            np.ndarray: (m, n, num_classes) boolean array with one channel filled with a True, all else False
        """
        max_ind = int(np.max(inds_image))
        num_classes = num_classes if num_classes is not None else max_ind + 1

        one_hot_array = np.zeros(
            (inds_image.shape[0], inds_image.shape[1], num_classes), dtype=bool
        )
        # Iterate up to max ind, not num_classes to avoid wasted computation when there won't be matches
        for i in range(max_ind + 1):
            # TODO determine if there are any more efficient ways to do this
            # Maybe create all these slices and then concatenate
            # Or test equality with an array that has all the values in it
            one_hot_array[..., i] = inds_image == i

        return one_hot_array


# class SegmentorPhotogrammetryCamera(PhotogrammetryCamera):
#    def __init__(self, base_camera: PhotogrammetryCamera, segmentor: Segmentor):
#        """Provide a segmented image instead of raw one
#
#        Args:
#            base_camera (PhotogrammetryCamera): The original camera
#            segmentor (Segmentor): The fully instantiated segmentor model
#        """
#        self.base_camera = base_camera
#        self.segmentor = segmentor
#
#        # This is ugly but I don't know how to do it better
#        # TODO look into composition instead of inheritence https://en.wikipedia.org/wiki/Composition_over_inheritance
#        # TODO look into python dataclasses https://docs.python.org/3/library/dataclasses.html
#        self.image_filename = base_camera.image_filename
#        self.cam_to_world_transform = base_camera.cam_to_world_transform
#        self.world_to_cam_transform = base_camera.world_to_cam_transform
#        self.f = base_camera.f
#        self.cx = base_camera.cx
#        self.cy = base_camera.cy
#        self.image_width = base_camera.image_width
#        self.image_height = base_camera.image_height
#        self.image_size = base_camera.image_size
#        self.image = base_camera.image
#        self.cache_image = base_camera.cache_image
#
#    def get_raw_image(self, image_scale: float = 1.0) -> np.ndarray:
#        """Gets the original image
#
#        Args:
#            image_scale (float, optional): scale factor per dimension. Defaults to 1.0.
#
#        Returns:
#            np.ndarray: The raw image
#        """
#        return self.base_camera.get_image(image_scale=image_scale)
#
#    def get_image(self, image_scale: float = 1) -> np.ndarray:
#        """Loads an image and segments it
#
#        Args:
#            image_scale (float, optional): Scale factor per dimension. Defaults to 1.
#
#        Returns:
#            np.ndarray: A one-hot encoded image
#        """
#        # TODO deterimine whether segmentation should be performed on the image
#        # at native resolution or the rescaled one. Or maybe flagged to provide both options
#
#        # For now, it's going to be at the rescaled one
#        raw_image = self.get_raw_image(image_scale=image_scale)
#        segmented_image = self.segmentor.segment_image(raw_image)
#        return segmented_image


class SegmentorPhotogrammetryCameraSet(PhotogrammetryCameraSet):
    def __init__(self, base_camera_set: PhotogrammetryCameraSet, segmentor: Segmentor):
        """Wraps a camera set to provide segmented versions of the image

        Args:
            base_camera_set (PhotogrammetryCameraSet): The original camera set
            segmentor (Segmentor): A fully instantiated segmentor
        """
        self.base_camera_set = base_camera_set
        self.segmentor = segmentor

        # This should allow all un-overridden methods to work as expected
        self.cameras = self.base_camera_set.cameras

    def get_image_by_index(self, index: int, image_scale: float = 1) -> np.ndarray:
        raw_image = self.base_camera_set.get_image_by_index(index, image_scale)
        segmented_image = self.segmentor.segment_image(raw_image)
        return segmented_image

    def get_raw_image_by_index(self, index: int, image_scale: float = 1) -> np.ndarray:
        return self.base_camera_set.get_image_by_index(
            index=index, image_scale=image_scale
        )

    def n_image_channels(self) -> int:
        return self.segmentor.num_classes
