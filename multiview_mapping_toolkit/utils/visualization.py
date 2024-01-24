from pathlib import Path
import numpy as np
from imageio import imread
import matplotlib.pyplot as plt

from multiview_mapping_toolkit.constants import NULL_TEXTURE_INT_VALUE


def visualize_renders(render_folder, image_folder, n_samples=5, render_extension="png"):
    rendered_files = np.random.choice(
        list(Path(render_folder).rglob(f"*.{render_extension}")), n_samples
    )

    for rendered_file in rendered_files:
        image_file = Path(
            image_folder, rendered_file.relative_to(render_folder)
        ).with_suffix(".JPG")
        image = imread(image_file)
        render = imread(rendered_file).astype(float)
        render[render == NULL_TEXTURE_INT_VALUE] = np.nan
        f, ax = plt.subplots(1, 2)
        ax[0].imshow(image)
        ax[1].imshow(render, vmin=0, vmax=9, cmap="tab10", interpolation="none")
        plt.show()
        plt.close()
