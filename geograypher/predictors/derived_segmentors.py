import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imageio import imread
from IPython.core.debugger import set_trace
from scipy.sparse import csr_array
from skimage.transform import resize

from geograypher.constants import PATH_TYPE
from geograypher.predictors import Segmentor


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
        if image_scale != 1:
            image = resize(
                image,
                (int(image.shape[0] * image_scale), int(image.shape[1] * image_scale)),
                order=0,  # Nearest neighbor interpolation
            )
        one_hot_image = self.inds_to_one_hot(image, num_classes=self.num_classes)
        return one_hot_image


class TabularRectangleSegmentor(Segmentor):
    def __init__(
        self,
        pred_file_or_folder,
        image_shape=(4008, 6016),
        label_key="label",
        image_path_key="image_path",
        imin_key="ymin",
        imax_key="ymax",
        jmin_key="xmin",
        jmax_key="xmax",
        predfile_extension="csv",
        strip_image_extension: bool = True,
        use_absolute_filepaths: bool = False,
        split_bbox: bool = True,
        image_folder=None,
    ):
        self.pred_file_or_folder = pred_file_or_folder
        self.image_shape = image_shape

        self.label_key = label_key
        self.image_path_key = image_path_key
        self.imin_key = imin_key
        self.imax_key = imax_key
        self.jmin_key = jmin_key
        self.jmax_key = jmax_key
        self.split_bbox = split_bbox

        self.predfile_extension = predfile_extension

        if os.path.isfile(pred_file_or_folder):
            files = [pred_file_or_folder]
        else:
            files = sorted(
                Path(self.pred_file_or_folder).glob("*" + self.predfile_extension)
            )

        dfs = [pd.read_csv(f) for f in files]

        self.labels_df = pd.concat(dfs, ignore_index=True)
        if "instance_ID" not in self.labels_df.columns:
            self.labels_df["instance_ID"] = self.labels_df.index

        if image_folder is not None and use_absolute_filepaths:
            absolute_filepaths = [
                str(Path(image_folder, img_path))
                for img_path in self.labels_df[self.image_path_key].tolist()
            ]
            self.labels_df[self.image_path_key] = absolute_filepaths

        if strip_image_extension:
            image_path_with_ext = [
                str(Path(img_path).with_suffix(""))
                for img_path in self.labels_df[self.image_path_key].tolist()
            ]
            self.labels_df[self.image_path_key] = image_path_with_ext

        self.grouped_labels_df = self.labels_df.groupby(by=self.image_path_key)
        self.image_names = list(self.grouped_labels_df.groups.keys())
        self.class_names = np.unique(self.labels_df[self.label_key]).tolist()
        self.num_classes = len(self.class_names)

        print(f"number of labeled images {len(self.image_names)}")

    def segment_image(self, image, filename, image_scale, vis=False):
        output_shape = self.image_shape
        label_image = np.full(output_shape, fill_value=np.nan, dtype=float)

        name = filename.name

        if name in self.image_names:
            df = self.grouped_labels_df.get_group(name)
        # Return an all-zero segmentation image
        else:
            return label_image

        for _, row in df.iterrows():
            label = row[self.label_key]
            label_ind = self.class_names.index(label)
            if self.split_bbox:
                # TODO split row
                bbox = row["bbox"]
                bbox = bbox[1:-1]
                splits = bbox.split(", ")
                jmin, imin, width, height = [float(s) for s in splits]

                imax = int(imin + height)
                jmax = int(jmin + width)

                imin = int(imin)
                jmin = int(jmin)
            else:
                imin = int(row[self.imin_key])
                imax = int(row[self.imax_key])
                jmin = int(row[self.jmin_key])
                jmax = int(row[self.jmax_key])

            label_image[imin:imax, jmin:jmax] = label_ind

        if vis:
            plt.imshow(label_image, vmin=0, vmax=10, cmap="tab10")
            plt.show()

        if image_scale != 1.0:
            output_size = (int(image_scale * x) for x in label_image.shape[:2])
            label_image = resize(label_image, output_size, order=0)

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
