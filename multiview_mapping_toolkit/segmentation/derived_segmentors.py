from pathlib import Path

import numpy as np
import pandas as pd
from imageio import imread
from skimage.transform import resize

from multiview_mapping_toolkit.config import PATH_TYPE
from multiview_mapping_toolkit.segmentation import Segmentor


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


class TabularRectangleSegmentor(Segmentor):
    def __init__(
        self,
        pred_folder,
        image_folder,
        image_shape=(4008, 6016),
        label_key="label",
        image_path_key="image_path",
        imin_key="ymin",
        imax_key="ymax",
        jmin_key="xmin",
        jmax_key="xmax",
        predfile_extension="csv",
    ):
        self.pred_folder = pred_folder
        self.image_folder = image_folder
        self.image_shape = image_shape

        self.label_key = label_key
        self.image_path_key = image_path_key
        self.imin_key = imin_key
        self.imax_key = imax_key
        self.jmin_key = jmin_key
        self.jmax_key = jmax_key

        self.predfile_extension = predfile_extension

        files = sorted(Path(self.pred_folder).glob("*" + self.predfile_extension))
        dfs = [pd.read_csv(f) for f in files]

        self.labels_df = pd.concat(dfs)
        self.grouped_labels_df = self.labels_df.groupby(by=self.image_path_key)
        self.image_names = list(self.grouped_labels_df.groups.keys())
        self.class_names = np.unique(self.labels_df[self.label_key]).tolist()
        self.num_classes = len(self.class_names)

    def segment_image(self, image, filename, image_scale, vis=False):
        label_image = np.zeros(
            self.image_shape + (len(self.class_names),), dtype=np.uint8
        )

        name = filename.name
        if name in self.image_names:
            df = self.grouped_labels_df.get_group(name)
        # Return an all-zero segmentation image
        else:
            return label_image

        for _, row in df.iterrows():
            label = row[self.label_key]
            label_ind = self.class_names.index(label)
            label_image[
                int(row[self.imin_key]) : int(row[self.imax_key]),
                int(row[self.jmin_key]) : int(row[self.jmax_key]),
                label_ind,
            ] = 1

        if vis:
            index_label = np.argmax(label_image, axis=2).astype(float)
            zeros_mask = np.sum(label_image, axis=2) == 0
            index_label[zeros_mask] = np.nan
            plt.imshow(index_label, vmin=0, vmax=10, cmap="tab10")
            plt.show()

        return label_image

    def get_detection_centers(self, filename):
        """_summary_

        Args:
            filename (_type_): _description_

        Returns:
            _type_: (n,2) array for (i,j) centers for each detection
        """
        if filename not in self.image_names:
            # Empty array of detection centers
            return np.zeros((0, 2))

        # Extract the corresponding dataframe
        df = self.grouped_labels_df.get_group(filename)

        # Extract the columns for the bounding box corners
        imin = df[self.imin_key].to_numpy()
        imax = df[self.imax_key].to_numpy()

        jmin = df[self.jmin_key].to_numpy()
        jmax = df[self.jmax_key].to_numpy()

        # Average the left-right, top-bottom pairs
        centers = np.vstack([(imin + imax) / 2, (jmin + jmax) / 2]).T
        return centers
