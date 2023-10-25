from rastervision.core.data import ClassConfig
from rastervision.core.data.label import SemanticSegmentationSmoothLabels
from rastervision.pytorch_learner import (
    SemanticSegmentationSlidingWindowGeoDataset,
)
from rastervision.core.evaluation import SemanticSegmentationEvaluator

from rastervision.core import Box
from tqdm import tqdm
from pathlib import Path
from tqdm import tqdm
from imageio import imwrite

import numpy as np

from multiview_prediction_toolkit.config import PATH_TYPE


class OrthoSegmentor:
    def __init__(
        self,
        raster_input_file: PATH_TYPE,
        vector_label_file: PATH_TYPE = None,
        raster_label_file: PATH_TYPE = None,
        class_names: list[str] = ("grass", "trees", "earth", "null"),
        class_colors: list[str] = ("lightgray", "darkred", "red", "green"),
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

        self.ds = None

    def get_filename(self, window):
        raster_name = Path(self.raster_input_file).name
        filename = (
            f"{raster_name}:{window.ymin}:{window.xmin}:{window.ymax}:{window.xmax}.png"
        )
        return filename

    def create_sliding_window_dataset(self, is_train: bool):
        kwargs = {
            "class_config": self.class_config,
            "image_uri": self.raster_input_file,
            "image_raster_source_kw": dict(allow_streaming=False),
            "size": self.chip_size,
        }

        if is_train:
            kwargs["label_vector_uri"] = self.vector_label_file
            kwargs["label_raster_uri"] = self.raster_label_file
            kwargs["stride"] = self.training_stride
        else:
            kwargs["stride"] = self.inference_stride

        self.ds = SemanticSegmentationSlidingWindowGeoDataset.from_uris(**kwargs)

    def create_training_chips(
        self,
        output_folder: PATH_TYPE,
        brightness_multiplier: float = 1.0,
        background_ind: int = 255,
    ):
        self.create_sliding_window_dataset(is_train=True)

        num_labels = len(self.class_config.names)

        # Create output folders
        image_folder = Path(output_folder, "imgs")
        anns_folder = Path(output_folder, "anns")

        image_folder.mkdir(parents=True, exist_ok=True)
        anns_folder.mkdir(parents=True, exist_ok=True)

        for (image, label), window in tqdm(
            zip(self.ds, self.ds.windows), total=len(self.ds)
        ):
            image = image.permute((1, 2, 0)).cpu().numpy()

            image = np.clip(image * 255 * brightness_multiplier, 0, 255).astype(
                np.uint8
            )

            label = label.cpu().numpy().astype(np.uint8)

            mask = label == (num_labels - 1)
            if np.all(mask):
                continue

            label[mask] = background_ind

            filename = self.get_filename(window)

            imwrite(Path(image_folder, filename), image)
            imwrite(Path(anns_folder, filename), label)

    def create_test_chips(
        self,
        output_folder: PATH_TYPE,
        brightness_multiplier: float = 1.0,
    ):
        self.create_sliding_window_dataset(is_train=False)

        Path(output_folder).mkdir(parents=True, exist_ok=True)

        for (image, _), window in tqdm(
            zip(self.ds, self.ds.windows), total=len(self.ds)
        ):
            image = image.permute((1, 2, 0)).cpu().numpy()

            image = np.clip(image * 255 * brightness_multiplier, 0, 255).astype(
                np.uint8
            )
            if np.all(image == 0):
                continue

            imwrite(Path(output_folder, self.get_filename(window=window)), image)

    def assemble_tiled_predictions(
        self,
        prediction_folder,
        savefile=None,
        discard_edge_frac=1 / 8,
        eval_performance=True,
    ):
        self.create_sliding_window_dataset(is_train=eval_performance)
        files = sorted(Path(prediction_folder).glob("*"))

        coords = [file.stem.split(":")[1:] for file in files]
        windows = [
            Box(
                xmin=int(coord[0]),
                ymin=int(coord[1]),
                xmax=int(coord[2]),
                ymax=int(coord[3]),
            )
            for coord in coords
        ]
        coords_array = np.array(coords).astype(int)
        extent = Box(
            xmin=np.min(coords_array[:, 0]),
            ymin=np.min(coords_array[:, 1]),
            xmax=np.max(coords_array[:, 2]),
            ymax=np.max(coords_array[:, 3]),
        )

        seg_labels = SemanticSegmentationSmoothLabels(
            extent=extent, num_classes=len(self.class_config.names) - 1, dtype=float
        )

        crop_sz = int(self.chip_size * discard_edge_frac)
        for window, file in tqdm(zip(windows, files), total=len(windows)):
            pred = np.load(file)
            pred = np.transpose(pred, (2, 0, 1))
            seg_labels.add_predictions(
                windows=[window], predictions=[pred], crop_sz=crop_sz
            )

        if savefile is not None:
            seg_labels.save(
                savefile,
                crs_transformer=self.ds.scene.raster_source.crs_transformer,
                class_config=self.class_config,
            )
        if eval_performance:
            evaluator = SemanticSegmentationEvaluator(self.class_config)
            gt_labels = self.ds.scene.label_source.get_labels()
            evaluation = evaluator.evaluate_predictions(
                ground_truth=gt_labels, predictions=seg_labels
            )
