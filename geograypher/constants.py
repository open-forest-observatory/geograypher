from pathlib import Path
from typing import Union

import matplotlib.colors
import matplotlib.pyplot as plt
import pyproj

## Typing constants
# A file/folder path
PATH_TYPE = Union[str, Path]

## Path constants
# Where all the data is stored
DATA_FOLDER = Path(Path(__file__).parent, "..", "data").resolve()
# Where to save vis data
VIS_FOLDER = Path(Path(__file__).parent, "..", "vis").resolve()
# Where to cache results
CACHE_FOLDER = Path(Path(__file__).parent, "..", "cache").resolve()

VERT_ID = "vert_ID"
CLASS_ID_KEY = "class_ID"
INSTANCE_ID_KEY = "instance_ID"
PRED_CLASS_ID_KEY = "pred_class_ID"
CLASS_NAMES_KEY = "class_names"
RATIO_3D_2D_KEY = "ratio_3d_2d"
NULL_TEXTURE_INT_VALUE = 255
LAT_LON_CRS = pyproj.CRS.from_epsg(4326)
EARTH_CENTERED_EARTH_FIXED_CRS = pyproj.CRS.from_epsg(4978)

### Example data variables
## Raw input data
# The input labels
EXAMPLE_SEGMENTATION_TASK_DATA = Path(DATA_FOLDER, "chips_tree_species")

# Name of the column to use in the example data
EXAMPLE_LABEL_COLUMN_NAME = "species_observed"
EXAMPLE_IDS_TO_LABELS = {
    0: "ABCO",
    1: "CADE",
    2: "PILA",
    3: "PIPJ",
    4: "PSME",
    5: "SNAG",
}
# The coordinate reference frame to interpret the mesh values in
EXAMPLE_MESH_CRS = 3310
EXAMPLE_LABELS_FILENAME = Path(
    EXAMPLE_SEGMENTATION_TASK_DATA, "inputs", "chips_ground_truth_labels.gpkg"
)
# The mesh exported from Metashape
EXAMPLE_MESH_FILENAME = Path(
    EXAMPLE_SEGMENTATION_TASK_DATA, "inputs", "chips_mesh_epsg3310_subset.ply"
)
# The camera file exported from Metashape
EXAMPLE_CAMERAS_FILENAME = Path(
    EXAMPLE_SEGMENTATION_TASK_DATA, "inputs", "chips_cameras.xml"
)
# The digital elevation map exported by Metashape
EXAMPLE_DTM_FILE = Path(EXAMPLE_SEGMENTATION_TASK_DATA, "inputs", "chips_DTM.tif")
# The image folder used to create the Metashape project
EXAMPLE_IMAGE_FOLDER = Path(EXAMPLE_SEGMENTATION_TASK_DATA, "inputs", "chips_images")
# The orthomosaic
EXAMPLE_ORTHO_FILENAME = Path(
    EXAMPLE_SEGMENTATION_TASK_DATA, "inputs", "chips_ortho.tif"
)

## Define the intermediate results
# Processed geo file
EXAMPLE_LABELED_MESH_FILENAME = Path(
    EXAMPLE_SEGMENTATION_TASK_DATA,
    "intermediate_results",
    "labeled_mesh.ply",
)
# Where to save the rendering label images
EXAMPLE_RENDERED_LABELS_FOLDER = Path(
    EXAMPLE_SEGMENTATION_TASK_DATA,
    "intermediate_results",
    "rendered_labels",
)
# Tiles from the orthomosaic
EXAMPLE_ORTHO_TILES_FOLDER = Path(
    EXAMPLE_SEGMENTATION_TASK_DATA, "intermediate_results", "ortho_tiles"
)
# Predicted images from a segementation algorithm
# TODO update this to real data
EXAMPLE_PREDICTED_LABELS_FOLDER = Path(
    EXAMPLE_SEGMENTATION_TASK_DATA,
    "intermediate_results",
    "image_predictions",
)
# The predicted aggregated face data
EXAMPLE_AGGREGATED_FACE_LABELS_FILE = Path(
    EXAMPLE_SEGMENTATION_TASK_DATA,
    "intermediate_results",
    "aggregated_face_labels.npy",
)

## Outputs
EXAMPLE_PREDICTED_VECTOR_LABELS_FILE = Path(
    EXAMPLE_SEGMENTATION_TASK_DATA,
    "outputs",
    "predicted_labels.geojson",
)

EXAMPLE_INTRINSICS = {
    "f": 1000,
    "cx": 0,
    "cy": 0,
    "image_width": 800,
    "image_height": 600,
    "distortion_params": {},
}


## Misc constants
# Colors to be used across the project
def hex_to_rgb(value):
    value = value.lstrip("#")
    lv = len(value)
    return tuple(int(value[i : i + lv // 3], 16) for i in range(0, lv, lv // 3))


MATPLOTLIB_PALLETE = [
    hex_to_rgb(x) for x in plt.rcParams["axes.prop_cycle"].by_key()["color"]
]
TEN_CLASS_VIS_KWARGS = {"cmap": "tab10", "clim": (-0.5, 9.5)}

DEFAULT_FRUSTUM_SCALE = 1
CHUNKED_MESH_BUFFER_DIST_METERS = 250
