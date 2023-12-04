import logging
import tempfile
import typing
from pathlib import Path

import numpy as np
import rasterio as rio
from imageio import imread, imwrite
from rastervision.core import Box
from rastervision.core.data import ClassConfig
from rastervision.core.data.label import (
    SemanticSegmentationDiscreteLabels,
    SemanticSegmentationSmoothLabels,
)
from rastervision.pytorch_learner import SemanticSegmentationSlidingWindowGeoDataset
from tqdm import tqdm

from multiview_prediction_toolkit.utils.io import read_image_or_numpy
from multiview_prediction_toolkit.config import MATPLOTLIB_PALLETE, PATH_TYPE
from multiview_prediction_toolkit.utils.numeric import create_ramped_weighting


class OrthoSegmentor:
    def __init__(
        self,
        raster_input_file: PATH_TYPE,
        vector_label_file: PATH_TYPE = None,
        raster_label_file: PATH_TYPE = None,
        class_names: list[str] = (),  # TODO fix these up
        class_colors: list[str] = (),  # TODO fix this
        chip_size: int = 2048,
        training_stride: int = 2048,
        inference_stride: int = 1024,
    ):
        self.raster_input_file = raster_input_file
        self.vector_label_file = vector_label_file
        self.raster_label_file = raster_label_file

        self.chip_size = chip_size
        self.training_stride = training_stride
        self.inference_stride = inference_stride

        self.class_names = class_names

        if class_colors is not None:
            class_colors = MATPLOTLIB_PALLETE[: len(class_names)]

        self.class_config = ClassConfig(
            names=class_names,
            colors=class_colors,
        )
        self.class_config.ensure_null_class()

        # This will be instantiated later
        self.dataset = None

    def get_filename_from_window(self, window: Box, extension: str = ".png") -> str:
        """Return a filename encoding the box coordinates

        Args:
            window (Box): Box to record
            extension (str, optional): Filename extension. Defaults to ".png".

        Returns:
            str: String with ymin, xmin, ymax, xmax
        """
        raster_name = Path(self.raster_input_file).name
        filename = f"{raster_name}:{window.ymin}:{window.xmin}:{window.ymax}:{window.xmax}{extension}"
        return filename

    def parse_windows_from_files(
        self, files: list[Path], sep: str = ":"
    ) -> tuple[list[rio.windows.Window], rio.windows.Window]:
        """Return the boxes and extent from a list of filenames

        Args:
            files (list[Path]): List of filenames
            sep (str): Seperator between elements

        Returns:
            tuple[list[rio.windows.Window], rio.windows.Window]: List of windows for each file and extent
        """
        # Split the coords out, currently ignorign the filename as the first element
        coords = [file.stem.split(sep)[1:] for file in files]
        # Create windows from coords
        windows = [
            rio.windows.Window(
                row_off=int(coord[0]),
                col_off=int(coord[1]),
                height=int(coord[2]) - int(coord[0]),
                width=int(coord[3]) - int(coord[1]),
            )
            for coord in coords
        ]
        # Compute the extents as the min/max of the boxes
        coords_array = np.array(coords).astype(int)

        xmin = np.min(coords_array[:, 0])
        ymin = np.min(coords_array[:, 1])
        xmax = np.max(coords_array[:, 2])
        ymax = np.max(coords_array[:, 3])
        extent = rio.windows.Window(
            row_off=ymin, col_off=xmin, width=xmax - xmin, height=ymax - ymin
        )

        return windows, extent

    def create_sliding_window_dataset(
        self, is_annotated: bool
    ) -> SemanticSegmentationSlidingWindowGeoDataset:
        """Sets self.dataset

        Args:
            is_annotated (bool): Should we use labels

        Returns:
            SemanticSegmentationSlidingWindowGeoDataset: The dataset, also sets self.dataset
        """

        # Keyword args shared between annotated and not
        kwargs = {
            "class_config": self.class_config,
            "image_uri": self.raster_input_file,
            "image_raster_source_kw": dict(allow_streaming=False),
            "size": self.chip_size,
        }

        if is_annotated:
            if self.raster_input_file is None and self.vector_label_file is None:
                raise ValueError("One type of label must be included")

            kwargs["label_vector_uri"] = self.vector_label_file
            kwargs["label_raster_uri"] = self.raster_label_file
            kwargs["stride"] = self.training_stride
        else:
            kwargs["stride"] = self.inference_stride

        # Create the dataset
        self.dataset = SemanticSegmentationSlidingWindowGeoDataset.from_uris(**kwargs)
        return self.dataset

    def write_training_chips(
        self,
        output_folder: PATH_TYPE,
        brightness_multiplier: float = 1.0,
        background_ind: int = 255,
        skip_all_nodata_tiles: bool = True,
    ):
        """Write out training tiles from raster

        Args:
            output_folder (PATH_TYPE): The folder to write to, will create 'imgs' and 'anns' subfolders if not present
            brightness_multiplier (float, optional): Multiply image brightness by this before saving. Defaults to 1.0.
            background_ind (int, optional): Write this value for unset/nodata pixels. Defaults to 255.
            skip_all_nodata_tiles (bool, optional): If a tile has no valid data, skip writing it
        """
        # Create annoated dataset
        self.create_sliding_window_dataset(is_annotated=True)

        num_labels = len(self.class_config.names)

        # Create output folders
        image_folder = Path(output_folder, "imgs")
        anns_folder = Path(output_folder, "anns")

        image_folder.mkdir(parents=True, exist_ok=True)
        anns_folder.mkdir(parents=True, exist_ok=True)

        # Main loop for writing out the data
        for (image, label), window in tqdm(
            zip(self.dataset, self.dataset.windows),
            total=len(self.dataset),
            desc="Saving train images and labels",
        ):
            label = label.cpu().numpy().astype(np.uint8)
            mask = label == (num_labels - 1)

            # If no data is valid, skip writing it
            if skip_all_nodata_tiles and np.all(mask):
                continue

            # Set the nodata regions to the background value
            label[mask] = background_ind

            # Traspose the image so it's channel-last, as expected for images on disk
            image = image.permute((1, 2, 0)).cpu().numpy()

            # Transform from 0-1 -> 0-255
            image = np.clip(image * 255 * brightness_multiplier, 0, 255).astype(
                np.uint8
            )

            # Get filename and write out
            filename = self.get_filename_from_window(window)
            imwrite(Path(image_folder, filename), image)
            imwrite(Path(anns_folder, filename), label)

    def write_inference_chips(
        self,
        output_folder: PATH_TYPE,
        brightness_multiplier: float = 1.0,
        skip_all_nodata_tiles: bool = True,
    ):
        """Writes out image chips to perform inference on

        Args:
            output_folder (PATH_TYPE): Where to write the image chips
            brightness_multiplier (float, optional): Multiply image brightness by this before saving. Defaults to 1.0.
            skip_all_nodata_tiles (bool, optional): Skip all black images. Defaults to True.
        """
        # Create dataset, without annotations
        self.create_sliding_window_dataset(is_annotated=False)

        # Create output folder
        Path(output_folder).mkdir(parents=True, exist_ok=True)

        # Iterate over images and windows in dataset
        for (image, _), window in tqdm(
            zip(self.dataset, self.dataset.windows),
            total=len(self.dataset),
            desc="Saving test images",
        ):
            # Transpose to channel-last
            image = image.permute((1, 2, 0)).cpu().numpy()
            # Brighten and transform from 0-1 -> 0-255
            image = np.clip(image * 255 * brightness_multiplier, 0, 255).astype(
                np.uint8
            )

            # Skip black images
            if skip_all_nodata_tiles and np.all(image == 0):
                continue

            # Write result
            imwrite(
                Path(output_folder, self.get_filename_from_window(window=window)), image
            )

    def assemble_tiled_predictions(
        self,
        prediction_folder: PATH_TYPE,
        savefile: typing.Union[PATH_TYPE, None] = None,
        discard_edge_frac: float = 1 / 8,
        eval_performance: bool = False,
        smooth_seg_labels: bool = False,
        nodataval: typing.Union[int, None] = 255,
        count_dtype: type = np.uint8,
        max_overlapping_tiles: int = 4,
    ):
        """Take tiled predictions on disk and aggregate them into a raster

        Args:
            pred_files (list[PATH_TYPE]): List of filenames where predictions are written
            class_savefile (typing.Union[PATH_TYPE, NoneType], optional): Where to save the merged raster.
            counts_savefile (typing.Union[PATH_TYPE, NoneType], optional):
                Where to save the counts for the merged predictions raster.
                A tempfile will be created and then deleted if not specified. Defaults to None.
            downweight_edge_frac (float, optional): Downweight this fraction of predictions at the edge of each tile using a linear ramp. Defaults to 0.25.
            smooth_seg_labels (bool, optional): Use a distribution of class confidences, rather than hard labels. Defaults to False.
            nodataval: (typing.Union[int, None]): Value for unassigned pixels. If None, will be set to len(self.class_names), the first unused class. Defaults to 255
            count_dtype (type, optional): What type to use for aggregation. Float uses more space but is more accurate. Defaults to np.uint8
            max_overlapping_tiles (int):
                The max number of prediction tiles that may overlap at a given point. This is used to upper bound the valud in the count matrix,
                because we use scaled np.uint8 values rather than floats for efficiency. Setting a lower value enables slightly more accuracy in the
                aggregation process, but too low can lead to overflow. Defaults to 4
        """
        # Setup
        num_classes = len(self.class_names)

        # Set nodataval to the first unused class ID
        if nodataval is None:
            nodataval = num_classes

        # If the user didn't specify where to write the counts, create a tempfile that will be deleted
        if counts_savefile is None:
            counts_savefile_manager = tempfile.NamedTemporaryFile(
                mode="w+", suffix=".tif", dir=class_savefile.parent
            )
            counts_savefile = counts_savefile_manager.name

        # Parse the filenames to get the windows
        # TODO consider using the extent to only write a file for the minimum encolsing rectangle
        windows, _ = self.parse_windows_from_files(pred_files)

        # Aggregate predictions
        with rio.open(self.raster_input_file) as src:
            # Create file to store counts that is the same as the input raster except it has num_classes number of bands
            with rio.open(
                counts_savefile,
                "w+",
                driver="GTiff",
                height=src.shape[0],
                width=src.shape[1],
                count=num_classes,
                dtype=count_dtype,
                crs=src.crs,
                transform=src.transform,
            ) as dst:
                # Create
                pred_weighting_dict = {}
                for pred_file, window in tqdm(
                    zip(pred_files, windows),
                    desc="Aggregating raster predictions",
                    total=len(pred_files),
                ):
                    # Read the prediction from disk
                    # TODO use more flexible reader here
                    pred = imread(pred_file)

                    if pred.shape != (window.height, window.width):
                        raise ValueError("Size of pred does not match window")

                    # We want to downweight portions at the edge so we create a ramped weighting mask
                    # but we don't want to duplicate this computation because it's the same for each same sized chip
                    if pred.shape not in pred_weighting_dict:
                        # We want to keep this as a uint8
                        pred_weighting = create_ramped_weighting(
                            pred.shape, downweight_edge_frac
                        )

                        # Allow us to get as much granularity as possible given the datatype
                        if count_dtype is not float:
                            pred_weighting = pred_weighting * (
                                np.iinfo(count_dtype).max / max_overlapping_tiles
                            )
                        # Convert weighting to desired type
                        pred_weighting_dict[pred.shape] = pred_weighting.astype(
                            count_dtype
                        )

                    # Get weighting
                    pred_weighting = pred_weighting_dict[pred.shape]

                    # Update each band in the counts file within the window
                    for i in range(num_classes):
                        # Bands in rasterio are 1-indexed
                        band_ind = i + 1
                        class_i_window_counts = dst.read(band_ind, window=window)
                        class_i_preds = pred == i
                        # If nothing matches this class, don't waste computation
                        if not np.any(class_i_preds):
                            continue
                        # Weight the predictions to downweight the ones at the edge
                        weighted_preds = (class_i_preds * pred_weighting).astype(
                            count_dtype
                        )
                        # Add the new predictions to the previous counts
                        class_i_window_counts += weighted_preds
                        # Write out the updated results for this window
                        dst.write(class_i_window_counts, band_ind, window=window)

        ## Convert counts file to max-class file

        with rio.open(counts_savefile, "r") as src:
            # Create a one-band file to store the index of the most predicted class
            with rio.open(
                class_savefile,
                "w",
                driver="GTiff",
                height=src.shape[0],
                width=src.shape[1],
                count=1,
                dtype=np.uint8,
                crs=src.crs,
                transform=src.transform,
                nodata=nodataval,
            ) as dst:
                # Iterate over the blocks corresponding to the tiff driver in the dataset
                # to compute the max class and write it out
                for _, window in tqdm(
                    list(src.block_windows()), desc="Writing out max class"
                ):
                    # Read in the counts
                    counts_array = src.read(window=window)
                    # Compute which pixels have no recorded predictions and mask them out
                    nodata_mask = np.sum(counts_array, axis=0) == 0

                    # If it's all nodata, don't write it out
                    # TODO make sure this works as expected
                    if np.all(nodata_mask):
                        continue

                    # Compute which class had the highest counts
                    max_class = np.argmax(counts_array, axis=0)
                    max_class[nodata_mask] = nodataval
                    # TODO, it would be good to check if it's all nodata and skip the write because that's unneeded
                    dst.write(max_class, 1, window=window)
