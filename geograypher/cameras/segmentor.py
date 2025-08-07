import typing
from copy import deepcopy

import numpy as np

from geograypher.cameras import PhotogrammetryCameraSet
from geograypher.predictors import Segmentor


class SegmentorPhotogrammetryCameraSet(PhotogrammetryCameraSet):
    def __init__(
        self,
        base_camera_set: PhotogrammetryCameraSet,
        segmentor: Segmentor,
        dont_load_base_image: bool = True,
    ):
        """Wraps a camera set to provide segmented versions of the image

        Args:
            base_camera_set (PhotogrammetryCameraSet): The original camera set
            segmentor (Segmentor): A fully instantiated segmentor
        """
        self.base_camera_set = base_camera_set
        self.segmentor = segmentor
        self.dont_load_base_image = dont_load_base_image

        # This should allow all un-overridden methods to work as expected
        self.cameras = self.base_camera_set.cameras
        self._local_to_epsg_4978_transform = (
            self.base_camera_set._local_to_epsg_4978_transform
        )

    def get_image_by_index(self, index: int, image_scale: float = 1) -> np.ndarray:
        if self.dont_load_base_image:
            raw_image = None
        else:
            raw_image = self.base_camera_set.get_image_by_index(index, image_scale)
        image_filename = self.base_camera_set.get_image_filename(index, absolute=True)
        segmented_image = self.segmentor.segment_image(
            raw_image, filename=image_filename, image_scale=image_scale
        )
        return segmented_image

    def get_raw_image_by_index(self, index: int, image_scale: float = 1) -> np.ndarray:
        return self.base_camera_set.get_image_by_index(
            index=index, image_scale=image_scale
        )

    def get_subset_cameras(self, inds: typing.List[int]):
        subset_camera_set = deepcopy(self)
        subset_camera_set.cameras = [subset_camera_set.cameras[i] for i in inds]
        subset_camera_set.base_camera_set = (
            subset_camera_set.base_camera_set.get_subset_cameras(inds)
        )
        return subset_camera_set

    def n_image_channels(self) -> int:
        return self.segmentor.num_classes

    def get_subset_with_valid_segmentation(self) -> "SegmentorPhotogrammetryCameraSet":
        """Get a new camera set consisting of all images that have a valid segmentation result

        Returns:
            SegmentorPhotogrammetryCameraSet: The subset of cameras with valid segmentation
        """
        valid_inds = []
        for i in range(len(self)):
            try:
                # Try to get the segmented result
                self.get_image_by_index(i)
                # If successful, append it to the list of valid IDs
                valid_inds.append(i)
            except:
                pass
        # Return the valid subset
        return self.get_subset_cameras(valid_inds)
