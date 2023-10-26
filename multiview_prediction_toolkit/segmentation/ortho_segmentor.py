from rastervision.core.data import ClassConfig
from rastervision.core.data.label import (
    SemanticSegmentationSmoothLabels,
    SemanticSegmentationDiscreteLabels,
)
from rastervision.pytorch_learner import (
    SemanticSegmentationSlidingWindowGeoDataset,
)
from rastervision.core.evaluation import SemanticSegmentationEvaluator

from rastervision.core import Box
from tqdm import tqdm
from pathlib import Path
from tqdm import tqdm
from imageio import imwrite
import typing

import numpy as np

from multiview_prediction_toolkit.config import PATH_TYPE


class OrthoSegmentor:
    def __init__(
        self,
        raster_input_file: PATH_TYPE,
        vector_label_file: PATH_TYPE = None,
        raster_label_file: PATH_TYPE = None,
        class_names: list[str] = (
            "grass",
            "trees",
            "earth",
            "null",
        ),  # TODO fix these up
        class_colors: list[str] = (
            "lightgray",
            "darkred",
            "red",
            "green",
        ),  # TODO fix this
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

        self.class_config = ClassConfig(
            names=class_names, colors=class_colors, null_class="null"
        )

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

    def parse_boxes_from_files(
        self, files: list[Path], sep: str = ":"
    ) -> tuple[list[Box], Box]:
        """Return the boxes and extent from a list of filenames

        Args:
            files (list[Path]): List of filenames
            sep (str): Seperator between elements

        Returns:
            tuple[list[Box], Box]: List of boxes for each file and extent
        """
        # Split the coords out, currently ignorign the filename as the first element
        coords = [file.stem.split(sep)[1:] for file in files]
        # Create windows from coords
        windows = [
            Box(
                xmin=int(coord[0]),
                ymin=int(coord[1]),
                xmax=int(coord[2]),
                ymax=int(coord[3]),
            )
            for coord in coords
        ]
        # Compute the extents as the min/max of the boxes
        coords_array = np.array(coords).astype(int)
        extent = Box(
            xmin=np.min(coords_array[:, 0]),
            ymin=np.min(coords_array[:, 1]),
            xmax=np.max(coords_array[:, 2]),
            ymax=np.max(coords_array[:, 3]),
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
        eval_performance: bool = True,
        smooth_seg_labels: bool = False,
    ):
        """Take tiled predictions on disk and aggregate them into a raster

        Args:
            prediction_folder (PATH_TYPE): Where predictions are written
            savefile (typing.Union[PATH_TYPE, NoneType], optional): Where to save the merged raster. Defaults to None.
            discard_edge_frac (float, optional): Discard this fraction of predictions at the edge of each tile. Defaults to 1/8.
            eval_performance (bool, optional): Compute metrics. Defaults to True.
            smooth_seg_labels (bool, optional): Use a distribution of class confidences, rather than hard labels. Defaults to False.
        """
        # Create dataset with annotation information if needed for evaluation
        self.create_sliding_window_dataset(is_annotated=eval_performance)
        # Get prediction files
        files = sorted(Path(prediction_folder).glob("*"))

        # Parse the filenames to get the windows
        windows, extent = self.parse_boxes_from_files(files)

        # Create the segmentation labels object to add predictions to
        seg_labels = (
            SemanticSegmentationSmoothLabels
            if smooth_seg_labels
            else SemanticSegmentationDiscreteLabels
        )(
            extent=extent,
            num_classes=len(self.class_config.names) - 1,
        )

        # Compute the number of pixels to discard at the edge
        crop_sz = int(self.chip_size * discard_edge_frac)
        # Iterate over windows and files
        for window, file in tqdm(
            zip(windows, files), total=len(windows), desc="Aggregating tile predictions"
        ):
            # TODO make this support more filetypes
            # Load the data
            pred = np.load(file)

            # This means the output is confidence-per-class, channel first
            if len(pred.shape) == 3:
                pred = np.transpose(pred, (2, 0, 1))

                # If we want hard classifications, compute that
                if not smooth_seg_labels:
                    pred = np.argmax(pred, axis=0)

            # Add the prediction to the dataset
            # TODO see if we can get speedups by batching this
            seg_labels.add_predictions(
                windows=[window], predictions=[pred], crop_sz=crop_sz
            )

        if savefile is not None:
            # Save out the raster data
            seg_labels.save(
                str(savefile),
                crs_transformer=self.dataset.scene.raster_source.crs_transformer,
                class_config=self.class_config,
            )

        # Compute the metrics
        if eval_performance:
            # TODO fix OOM issues here
            evaluator = SemanticSegmentationEvaluator(self.class_config)
            gt_labels = self.dataset.scene.label_source.get_labels()
            evaluation = evaluator.evaluate_predictions(
                ground_truth=gt_labels, predictions=seg_labels
            )
            print(evaluation)
            return evaluation
