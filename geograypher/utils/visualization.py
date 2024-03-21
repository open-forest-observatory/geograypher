from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from imageio import imread

from geograypher.constants import NULL_TEXTURE_INT_VALUE, TEN_CLASS_VIS_KWARGS


def show_segmentation_labels(
    label_folder,
    image_folder,
    null_label=NULL_TEXTURE_INT_VALUE,
    imshow_kwargs=TEN_CLASS_VIS_KWARGS,
    num_show=10,
    label_suffix=".png",
    image_suffix=".JPG",
):
    rendered_files = list(Path(label_folder).rglob("*" + label_suffix))
    np.random.shuffle(rendered_files)

    for rendered_file in rendered_files[:num_show]:
        image_file = Path(
            image_folder, rendered_file.relative_to(label_folder)
        ).with_suffix(image_suffix)

        image = imread(image_file)
        render = imread(rendered_file).astype(float)
        render[render == null_label] = np.nan
        _, ax = plt.subplots(1, 2)
        ax[0].imshow(image)
        ax[1].imshow(
            render,
            interpolation="none",
            cmap=imshow_kwargs["cmap"],
            vmin=imshow_kwargs["clim"][0],
            vmax=imshow_kwargs["clim"][1],
        )
        plt.show()
        plt.close()
