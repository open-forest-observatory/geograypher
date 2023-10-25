from operator import is_
from rastervision.core.data import ClassConfig
from rastervision.core.data.label import SemanticSegmentationSmoothLabels
from rastervision.pytorch_learner import (
    SemanticSegmentationSlidingWindowGeoDataset,
    SemanticSegmentationVisualizer,
)
from rastervision.core.evaluation import SemanticSegmentationEvaluator

from rastervision.core import Box
from imageio import imsave
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from tqdm import tqdm

import albumentations as A
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

    def create_training_chips(self):
        self.create_sliding_window_dataset(is_train=True)

    def create_test_chips(self):
        self.create_sliding_window_dataset(is_train=False)

    def assemble_tiled_predictions(self, prediction_folder, savefile, crop_frac=1 / 8):
        self.create_sliding_window_dataset(is_train=False)
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
        gt_labels = self.ds.get_label_array()
        breakpoint()

        crop_sz = int(self.chip_size * crop_frac)
        for window, file in tqdm(zip(windows, files), total=len(windows)):
            pred = np.load(file)
            pred = np.transpose(pred, (2, 0, 1))
            seg_labels.add_predictions(
                windows=[window], predictions=[pred], crop_sz=crop_sz
            )

        seg_labels.save(
            savefile,
            crs_transformer=self.ds.scene.raster_source.crs_transformer,
            class_config=self.class_config,
        )


IMAGE_URI = "data/gascola/orthos/example-run-001_20230517T1827_ortho_mesh.tif"
LABELS_URI = "data/gascola/gascola.geojson"


ortho_seg = OrthoSegmentor(raster_input_file=IMAGE_URI, vector_label_file=LABELS_URI)
for tag in ("001", "002"):
    pred_folder = f"data/gascola/ortho_chips_saved_pred_{tag}"
    savefile = f"data/gascola/ortho_chips_saved_pred_{tag}.tif"
    ortho_seg.assemble_tiled_predictions(pred_folder, savefile)
