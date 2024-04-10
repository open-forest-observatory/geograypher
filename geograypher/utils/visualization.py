import typing
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from imageio import imread, imwrite

from geograypher.constants import (
    NULL_TEXTURE_INT_VALUE,
    PATH_TYPE,
    TEN_CLASS_VIS_KWARGS,
)
from geograypher.utils.files import ensure_folder


def create_composite(
    RGB_image: np.ndarray,
    label_image: np.ndarray,
    label_blending_weight: float = 0.5,
    IDs_to_labels: typing.Union[None, dict] = None,
):
    """Create a three-panel composite with an RGB image and a label

    Args:
        RGB_image (np.ndarray):
            (h, w, 3) rgb image to be used directly as one panel
        label_image (np.ndarray):
            (h, w) image containing either integer labels or float scalars. Will be colormapped
            prior to display.
        label_blending_weight (float, optional):
            Opacity for the label in the blended composite. Defaults to 0.5.
        IDs_to_labels (typing.Union[None, dict], optional):
            Mapping from integer IDs to string labels. Used to compute colormap. If None, a
            continous colormap is used. Defaults to None.

    Raises:
        ValueError: If the RGB image cannot be interpreted as such

    Returns:
        np.ndarray: (h, 3*w, 3) horizontally composited image
    """
    if RGB_image.ndim != 3 or RGB_image.shape[2] != 3:
        raise ValueError("Invalid RGB error")

    if RGB_image.dtype == np.uint8:
        # Rescale to float range and implicitly cast
        RGB_image = RGB_image / 255

    if not (label_image.ndim == 3 and label_image.shape[2] == 3):
        null_mask = label_image == NULL_TEXTURE_INT_VALUE
        if IDs_to_labels is not None:
            # This produces a float colormapped values based on the indices
            label_image = plt.cm.tab10(label_image)[..., :3]
        else:
            # TODO this should be properly scaled
            label_image = plt.cm.viridis(label_image)[..., :3]
        label_image[null_mask] = 0
    # Determine if the label image needs to be colormapped into a 3 channel image

    overlay = ((1 - label_blending_weight) * RGB_image) + (
        label_blending_weight * label_image
    )
    composite = np.concatenate((label_image, RGB_image, overlay), axis=1)
    # Cast to np.uint8 for saving
    composite = (composite * 255).astype(np.uint8)
    return composite


def show_segmentation_labels(
    label_folder,
    image_folder,
    savefolder: typing.Union[None, PATH_TYPE] = None,
    null_label=NULL_TEXTURE_INT_VALUE,
    imshow_kwargs=TEN_CLASS_VIS_KWARGS,
    num_show=10,
    label_suffix=".png",
    image_suffix=".JPG",
):
    rendered_files = list(Path(label_folder).rglob("*" + label_suffix))
    np.random.shuffle(rendered_files)

    if savefolder is not None:
        ensure_folder(savefolder)

    for i, rendered_file in enumerate(rendered_files[:num_show]):
        image_file = Path(
            image_folder, rendered_file.relative_to(label_folder)
        ).with_suffix(image_suffix)

        image = imread(image_file)
        render = imread(rendered_file)
        composite = create_composite(image, render)

        if savefolder is None:
            plt.imshow(composite)
            plt.show()
        else:
            output_file = Path(savefolder, f"rendered_label_{i:03}.png")
            imwrite(output_file, composite)
