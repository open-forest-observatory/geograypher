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
PRED_CLASS_ID_KEY = "pred_class_ID"
CLASS_NAMES_KEY = "class_names"
RATIO_3D_2D_KEY = "ratio_3d_2d"
NULL_TEXTURE_INT_VALUE = 255
LAT_LON_EPSG_CODE = pyproj.CRS.from_epsg(4326)
EARTH_CENTERED_EARTH_FIXED_EPSG_CODE = pyproj.CRS.from_epsg(4978)

### Example data variables
## Raw input data
# The input labels
EXAMPLE_LABELS_FILENAME = Path(
    DATA_FOLDER, "example_Emerald_Point_data", "inputs", "labels.geojson"
)
EXAMPLE_IDS_TO_LABELS = {
    0: "ABCO",
    1: "ABMA",
    2: "CADE",
    3: "PI",
    4: "PICO",
    5: "PIJE",
    6: "PILA",
    7: "PIPO",
    8: "SALSCO",
    9: "TSME",
}
# The mesh exported from Metashape
EXAMPLE_MESH_FILENAME = Path(
    DATA_FOLDER, "example_Emerald_Point_data", "inputs", "mesh.ply"
)
# The camera file exported from Metashape
EXAMPLE_CAMERAS_FILENAME = Path(
    DATA_FOLDER, "example_Emerald_Point_data", "inputs", "cameras.xml"
)
# The digital elevation map exported by Metashape
EXAMPLE_DTM_FILE = Path(DATA_FOLDER, "example_Emerald_Point_data", "inputs", "dtm.tif")
# The image folder used to create the Metashape project
EXAMPLE_IMAGE_FOLDER = Path(
    DATA_FOLDER, "example_Emerald_Point_data", "inputs", "images"
)

## Define the intermediate results
# Processed geo file
EXAMPLE_STANDARDIZED_LABELS_FILENAME = Path(
    DATA_FOLDER,
    "example_Emerald_Point_data",
    "intermediate_results",
    "standardized_labels.geojson",
)
# Where to save the mesh after labeling
EXAMPLE_LABELED_MESH_FILENAME = Path(
    DATA_FOLDER,
    "example_Emerald_Point_data",
    "intermediate_results",
    "labeled_mesh.ply",
)
# Where to save the rendering label images
EXAMPLE_RENDERED_LABELS_FOLDER = Path(
    DATA_FOLDER, "example_Emerald_Point_data", "intermediate_results", "rendered_labels"
)
# Predicted images from a segementation algorithm
EXAMPLE_PREDICTED_LABELS_FOLDER = Path(
    DATA_FOLDER,
    "example_Emerald_Point_data",
    "intermediate_results",
    "predicted_segmentations",
)
# The predicted aggregated face data
EXAMPLE_AGGREGATED_FACE_LABELS_FILE = Path(
    DATA_FOLDER,
    "example_Emerald_Point_data",
    "intermediate_results",
    "aggregated_face_labels.npy",
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
TWENTY_CLASS_VIS_KWARGS = {"cmap": "tab20", "clim": (-0.5, 19.5)}

## Outputs
EXAMPLE_PREDICTED_VECTOR_LABELS_FILE = Path(
    DATA_FOLDER,
    "example_Emerald_Point_data",
    "outputs",
    "predicted_labels.geojson",
)

EXAMPLE_LABEL_NAMES = (
    "ABCO",
    "ABMA",
    "CADE",
    "PI",
    "PICO",
    "PIJE",
    "PILA",
    "PIPO",
    "SALSCO",
    "TSME",
)

DEFAULT_FRUSTUM_SCALE = 1
