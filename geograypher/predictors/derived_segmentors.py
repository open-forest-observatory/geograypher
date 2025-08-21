import os
import typing
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imageio import imread
from PIL import Image
from skimage import draw
from skimage.transform import resize

from geograypher.constants import PATH_TYPE
from geograypher.predictors import Segmentor
from geograypher.utils.files import ensure_containing_folder


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


class ImageIDSegmentor(Segmentor):
    """Setmentor that returns an image full of the index of that camera in the set"""

    def __init__(self, image_filenames: typing.List[PATH_TYPE]):
        """Set up the segmentor based on a list of image filenames.

        Args:
            image_filenames (typing.List[PATH_TYPE]):
                The list of absolute image paths. In the segmentation stage, the returned value with
                be the index of an image within this list.
        """
        self.image_filenames = image_filenames

    def segment_image(self, image: np.ndarray, filename: PATH_TYPE, image_scale: float):
        # Get the shape of the image without reading it into memory
        img_handler = Image.open(filename)
        w, h = img_handler.size

        # Get the index of the image in the list
        image_index = self.image_filenames.index(filename)
        if image_index is None:
            raise ValueError(f"Image {filename} not found in list")

        # Get the scaled shape to output
        output_shape = (int(h * image_scale), int(w * image_scale))
        # Create an array of the appropriate size filled with the value of the index within the list
        ID_image = np.full(output_shape, fill_value=image_index, dtype=int)
        return ID_image


class TabularRectangleSegmentor(Segmentor):
    def __init__(
        self,
        detection_file_or_folder: PATH_TYPE,
        image_shape: tuple,
        label_key: str = "instance_ID",
        image_path_key: str = "image_path",
        imin_key: str = "ymin",
        imax_key: str = "ymax",
        jmin_key: str = "xmin",
        jmax_key: str = "xmax",
        detection_file_extension: str = "csv",
        strip_image_extension: bool = False,
        use_absolute_filepaths: bool = False,
        split_bbox: bool = True,
        image_folder: typing.Union[PATH_TYPE, None] = None,
    ):
        """Lookup rectangular bounding boxes corresponding to detections from a CSV or folder of them.

        Args:
            detection_file_or_folder (PATH_TYPE):
                Path to the CSV file with detections or a folder thereof
            image_shape (tuple):
                The (height, width) shape of the image in pixels.
            label_key (str, optional):
                The column that corresponds to the class. Defaults to "label".
            image_path_key (str, optional):
                The column that has the image filename. Defaults to "image_path".
            imin_key (str, optional):
                Column of the minimum i dimension. Defaults to "ymin".
            imax_key (str, optional):
                Column of the max i dimension. Defaults to "ymax".
            jmin_key (str, optional):
                Column of the min j dimension. Defaults to "xmin".
            jmax_key (str, optional):
                Column of the max j dimension. Defaults to "xmax".
            detection_file_extension (str, optional):
                File extension of the detection files. Defaults to "csv".
            strip_image_extension (bool, optional):
                Remove the extension from the image filenames. Defaults to True.
            use_absolute_filepaths (bool, optional):
                Add the absolute path from the image folder to the filenames. Defaults to False.
            split_bbox (bool, optional):
                Split the bounding box from one column rather than having seperate columns for imin,
                imax, jmin, jmax. Defaults to True.
            image_folder (PATH_TYPE, optional): Path to the image folder. Defaults to None.
        """
        self.image_shape = image_shape

        self.label_key = label_key
        self.image_path_key = image_path_key
        self.imin_key = imin_key
        self.imax_key = imax_key
        self.jmin_key = jmin_key
        self.jmax_key = jmax_key
        self.split_bbox = split_bbox

        # Load the detections
        self.labels_df = self.load_detection_files(
            detection_file_or_folder=detection_file_or_folder,
            detection_file_extension=detection_file_extension,
            image_folder=image_folder,
            use_absolute_filepaths=use_absolute_filepaths,
            strip_image_extension=strip_image_extension,
            image_path_key=image_path_key,
        )

        # Group the predictions
        self.grouped_labels_df = self.labels_df.groupby(by=self.image_path_key)

        # List the images
        self.image_names = list(self.grouped_labels_df.groups.keys())
        # Record the class names and number of classes
        self.class_names = np.unique(self.labels_df[self.label_key]).tolist()
        self.num_classes = len(self.class_names)

    def load_detection_files(
        self,
        detection_file_or_folder: PATH_TYPE,
        detection_file_extension: str,
        image_folder: PATH_TYPE,
        use_absolute_filepaths: bool,
        strip_image_extension: bool,
        image_path_key: str,
    ):
        # Determine whether the input is a file or folder
        if Path(detection_file_or_folder).is_file():
            # If it's a file, make a one-length list
            files = [detection_file_or_folder]
        else:
            # List all the files in the folder with the requested extesion
            files = sorted(
                Path(detection_file_or_folder).glob("*" + detection_file_extension)
            )

        # Read the individual files
        dfs = [pd.read_csv(f) for f in files]

        # Concatenate the dataframes into one
        labels_df = pd.concat(dfs, ignore_index=True)

        # Add an sequential instance ID column if not present
        if "instance_ID" not in labels_df.columns:
            labels_df["instance_ID"] = labels_df.index

        # Prepend the image folder to the image filenames if requested to make an absolute filepath
        if image_folder is not None and use_absolute_filepaths:
            absolute_filepaths = [
                str(Path(image_folder, img_path))
                for img_path in labels_df[image_path_key].tolist()
            ]
            labels_df[image_path_key] = absolute_filepaths

        # Strip the extension from the image filenames if requested
        if strip_image_extension:
            image_path_without_ext = [
                str(Path(img_path).with_suffix(""))
                for img_path in labels_df[image_path_key].tolist()
            ]
            labels_df[image_path_key] = image_path_without_ext

        return labels_df

    def get_all_detections(self) -> pd.DataFrame:
        """Return the aggregated detections dataframe"""
        return self.labels_df

    def save_detection_data(self, output_csv_file: PATH_TYPE):
        """Save the aggregated detections to a file

        Args:
            output_csv_file (PATH_TYPE):
                A path to a CSV file to save the detections to. The containing folder will be
                created if needed.
        """
        ensure_containing_folder(output_csv_file)
        self.labels_df.to_csv(output_csv_file)

    def get_corners(self, data, as_int=True):
        if self.split_bbox:
            # TODO split row
            bbox = data["bbox"]
            bbox = bbox[1:-1]
            splits = bbox.split(", ")
            jmin, imin, width, height = [float(s) for s in splits]

            imax = imin + height
            jmax = jmin + width

            imin = imin
            jmin = jmin
        else:
            imin = data[self.imin_key]
            imax = data[self.imax_key]
            jmin = data[self.jmin_key]
            jmax = data[self.jmax_key]

        corners = imin, jmin, imax, jmax
        if as_int:
            corners = list(map(int, corners))

        return corners

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

            imin, jmin, imax, jmax = self.get_corners(
                row,
            )

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

        all_corners = []
        for _, row in df.iterrows():
            corners = self.get_corners(row, as_int=False)
            all_corners.append(corners)

        all_corners = zip(*all_corners)
        all_corners = [np.array(x) for x in all_corners]

        imin, jmin, imax, jmax = all_corners

        # Average the left-right, top-bottom pairs
        centers = np.vstack([(imin + imax) / 2, (jmin + jmax) / 2]).T
        return centers


class RegionDetectionSegmentor(Segmentor):
    def __init__(
        self,
        base_folder: PATH_TYPE,
        lookup_folder: PATH_TYPE,
        label_key: str,
        class_map: dict,
        geo_file_extension: str = ".gpkg",
    ):
        """Lookup region detections from geospatial vector files (such as .gpkg,
        .geojson, or .shp) files using geopandas. The lookup process will start with
        an image name, which will then be used to search for the proper detection file.

        Assumes that each geospatial filepath matches the corresponding image path
        (with different extension, and relative to a different root directory).

        Args:
            base_folder (PATH_TYPE):
                Path to the root directory for the images. There may be images in this
                directory, or a nested set of folders with images inside.
            lookup_folder (PATH_TYPE):
                Path to the root directory for the geospatial files with detections. The
                nested geospatial folder/file structure should match that of the images in
                the base folder.
            label_key (str):
                The column in the geospatial dataframe (from the files in lookup_folder)
                that corresponds to the class. For example, if half of the detections are
                from tree species A and half are from tree species B, then the gdf[label_key]
                should store that information. This is used in segment_image so that different
                channels are created for each defined class. If you want a class per detection,
                then make a column with a unique value per row and use that as the labels.
            class_map (dict):
                Maps from labels (see argument above) to integer indices that can be used to
                index into a (H, W, N classes) array. For example, if half of the detections are
                from tree species A, then the mapping should include something like {"A": 0},
                where segmented_image[:, :, 0] contains all the detections for that label.
            geo_file_extension (str):
                The file extension for the image files (e.g., ".gpkg", "geojson", ".shp").
                Defaults to ".gpkg".
        """
        self.base_folder = Path(base_folder)
        self.lookup_folder = Path(lookup_folder)
        self.geo_file_extension = geo_file_extension

        # Save these for segment_image
        self.label_key = label_key
        self.class_map = class_map

        # Do some input checking
        if not self.lookup_folder.is_dir():
            raise ValueError(f"Folder {self.lookup_folder} not found")

    def geomatch(self, impath):
        """Helper function to find a matching geospatial file for an image."""
        # Find image path relative to the root
        subpath = Path(impath).relative_to(self.base_folder)
        # Construct a geospatial file with the same subpath
        return self.lookup_folder / subpath.with_suffix(self.geo_file_extension)

    def get_detection_centers(self, im_path: PATH_TYPE) -> np.ndarray:
        """Get the centers of all detections for a given image filepath.

        Args:
            im_path: The image filepath to get detection centers for.

        Returns:
            np.ndarray: (n,2) array for (i,j) centers for each detection
        """
        geo_path = self.geomatch(im_path)
        if not geo_path.is_file():
            # Empty array of detection centers
            return np.zeros((0, 2))

        # Extract the corresponding geodataframe
        gdf = gpd.read_file(geo_path)

        # Calculate (N, 2) centers from geometry centroids
        return np.vstack([gdf.centroid.y, gdf.centroid.x]).T

    def segment_image(
        self, image: None, im_path: PATH_TYPE, image_shape: tuple
    ) -> np.ndarray:
        """
        Produce a segmentation mask for an image using region (polygon) detections.
        Each region will be checked for it's class label (e.g. tree species or some other
        label) using the [self.labels_key] column in the geospatial file. Then the
        integer index will be calculated from self.class_map[label]

        Args:
            image: The input image array (not used for region lookup, but kept for
                API compatibility).
            im_path: The image filepath to look up regions for.
            image_shape: (2,) tuple of the (height, width) of the output mask we want

        Returns:
            one_hot_labels: A 3D one-hot numpy array of shape (H, W, N classes). For a
                given polygon index, the [:, :, index] slice will be True where the
                mask is present and False otherwise. dtype=bool. If no image match is
                present, return a (H, W, 0) mask.
        """

        geo_path = self.geomatch(im_path)
        if not geo_path.is_file():
            return np.full(image_shape + (0,), fill_value=False, dtype=bool)
        gdf = gpd.read_file(geo_path)

        # Do some input checking
        if self.label_key not in gdf.columns:
            raise ValueError(
                f"label key ({self.label_key}) not found in GDF columns:\n{gdf.columns}"
            )
        if len(difference := set(gdf[self.label_key]) - set(self.class_map.keys())) > 0:
            raise ValueError(
                "Found the following label keys in a GDF which were not in the class"
                f" map: {difference}"
            )
        if any([not isinstance(value, int) for value in self.class_map.values()]):
            raise ValueError(
                "Found class map values which were not integer indices:\n"
                f"{self.class_map.values()}"
            )

        # Store the polygon masks in a (H, W, N classes) array
        num_classes = max(self.class_map.values()) + 1
        label_image = np.full(
            image_shape + (num_classes,), fill_value=False, dtype=bool
        )

        # Bookkeep which polygon we are currently drawing, in case the GDF has non
        # polygonal rows that get skipped.
        for _, row in gdf.iterrows():

            # Collect all polygons (whether single or multi) into a list
            if row.geometry.geom_type == "Polygon":
                polygons = [row.geometry]
            elif row.geometry.geom_type == "MultiPolygon":
                polygons = list(row.geometry.geoms)
            else:
                continue

            # Get the index by checking the label column against the class map
            index = self.class_map[row[self.label_key]]

            # Draw the polygons. Note that this closes internal holes because we are
            # using poly.exterior.
            for poly in polygons:
                # Note: (y, x) because draw.polygon uses row, col
                y, x = poly.exterior.xy
                rows, cols = draw.polygon(
                    np.array(y), np.array(x), shape=label_image.shape
                )
                label_image[rows, cols, index] = True

        return label_image
