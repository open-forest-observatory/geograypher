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
        detection_file_or_folder: PATH_TYPE,
        geo_file_extension: str = ".gpkg",
        image_file_extension: str = ".JPG",
    ):
        """Lookup region detections from a geospatial vector file (such as .gpkg,
        .geojson, or .shp) files using geopandas.

        Assumes that each .gpkg filename matches the corresponding image filename
        (with different extension).

        Args:
            detection_file_or_folder (PATH_TYPE):
                Path to the .gpkg file with detections or a folder thereof
            geo_file_extension (str, optional):
                The file extension for the image files (e.g., ".gpkg", "geojson", ".shp").
                Defaults to ".gpkg".
            image_file_extension (str, optional):
                The file extension for the image files (e.g., ".JPG", ".png"). Defaults to ".JPG".
        """
        self.geo_file_extension = geo_file_extension
        self.image_file_extension = image_file_extension

        # Load the detections
        self.labels_gdf = self.load_detection_files(detection_file_or_folder)

        # Group the predictions by image name (derived from .gpkg filename)
        self.grouped_labels_gdf = self.labels_gdf.groupby(by="image_name")

        # List the images
        self.image_names = list(self.grouped_labels_gdf.groups.keys())

    def load_detection_files(self, detection_file_or_folder: PATH_TYPE) -> pd.DataFrame:

        # Determine whether the input is a file or folder
        if Path(detection_file_or_folder).is_file():
            # If it's a file, make a one-length list
            files = [detection_file_or_folder]
        else:
            # List all geodata files in the folder
            files = sorted(
                Path(detection_file_or_folder).glob(f"*{self.geo_file_extension}")
            )

        # Read the individual files using geopandas and add image name column
        gdfs = []
        for f in files:
            gdf = gpd.read_file(f)
            # Extract image name from .gpkg filename and add the image extension
            image_name = f.stem + self.image_file_extension

            # Add image name column to the geodataframe
            gdf["image_name"] = image_name
            gdfs.append(gdf)

        # Concatenate the geodataframes into one
        if len(gdfs) > 0:
            labels_gdf = pd.concat(gdfs, ignore_index=True)
        else:
            # Empty dataframe
            labels_gdf = gpd.GeoDataFrame(
                columns=["image_name", "geometry"], geometry="geometry"
            )

        return labels_gdf

    def get_detection_centers(self, filename: str) -> np.ndarray:
        """Get the centers of all detections for a given image filename.

        Args:
            filename: The image filename to get detection centers for.

        Returns:
            np.ndarray: (n,2) array for (i,j) centers for each detection
        """
        if filename not in self.image_names:
            # Empty array of detection centers
            return np.zeros((0, 2))

        # Extract the corresponding geodataframe
        gdf = self.grouped_labels_gdf.get_group(filename)

        # Calculate (N, 2) centers from geometry centroids
        return np.vstack([gdf.centroid.x, gdf.centroid.y]).T

    def segment_image(
        self, image: None, filename: str, image_shape: tuple
    ) -> np.ndarray:
        """
        Produce a segmentation mask for an image using region (polygon) detections.
        Note that since region detections can overlap but segmentation masks cannot,
        the order of the regions will be respected and so later regions will be placed
        on top of conflicting earlier regions. Each region will be assigned a unique
        integer label corresponding to its order in the file (0, 1, 2, ...).

        Args:
            image: The input image array (not used for region lookup, but kept for
                API compatibility).
            filename: The image filename to look up regions for.
            image_shape: (2,) tuple of the (height, width) of the output mask we want

        Returns:
            label_image: A 2D numpy array of shape (H, W) with integer region indices for each pixel.
                         Pixels not covered by any region are set to np.nan.
        """
        label_image = np.full(image_shape, fill_value=np.nan, dtype=float)

        if filename not in self.image_names:
            return label_image

        gdf = self.grouped_labels_gdf.get_group(filename)

        for label, (_, row) in enumerate(gdf.iterrows()):

            # Collect all polygons (whether single or multi) into a list
            if row.geometry.geom_type == "Polygon":
                polygons = [row.geometry]
            elif row.geometry.geom_type == "MultiPolygon":
                polygons = list(row.geometry.geoms)
            else:
                continue

            for poly in polygons:
                # Note: (y, x) because draw.polygon uses row, col
                y, x = poly.exterior.xy
                rows, cols = draw.polygon(
                    np.array(y), np.array(x), shape=label_image.shape
                )
                label_image[rows, cols] = label

        return label_image
