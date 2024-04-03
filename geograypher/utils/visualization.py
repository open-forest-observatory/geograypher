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


def create_composite(RGB_image, label_image, label_weight=0.5):
    if RGB_image.ndim != 3 or RGB_image.shape[2] != 3:
        raise ValueError("Invalid RGB error")

    if RGB_image.dtype == np.uint8:
        # Rescale to float range and implicitly cast
        RGB_image = RGB_image / 255

    if not (label_image.ndim == 3 and label_image.shape[2] == 3):
        if label_image.dtype == np.uint8:
            null_mask = label_image == NULL_TEXTURE_INT_VALUE
            # This produces a float colormapped values based on the indices
            label_image = plt.cm.tab10(label_image)[..., :3]
            label_image[null_mask] = 0
        else:
            print("continous")
            breakpoint()
    # Determine if the label image needs to be colormapped into a 3 channel image

    overlay = ((1 - label_weight) * RGB_image) + (label_weight * label_image)
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
