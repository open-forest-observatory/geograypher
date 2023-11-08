from pathlib import Path
from typing import Union

## Typing constants
# A file/folder path
PATH_TYPE = Union[str, Path]

## Path constants
# Where all the data is stored
DATA_FOLDER = Path(Path(__file__).parent, "..", "data").resolve()
# Where to save vis data
VIS_FOLDER = Path(Path(__file__).parent, "..", "vis").resolve()

# Images for one metashape project used as an example
DEFAULT_IMAGES_FOLDER = str(Path(DATA_FOLDER, "composite_georef", "images"))

# Images for one metashape project used as an example
DEFAULT_LABELS_FOLDER = str(Path(DATA_FOLDER, "composite_georef", "segmented"))

# Mesh in local metashape coordinates for one metashape project used as an example
DEFAULT_LOCAL_MESH = str(
    Path(
        DATA_FOLDER,
        "composite_georef",
        "composite_georef_low_res_20230928T1604_model_local.ply",
    )
)
# Exported metashape cameras for example project
DEFAULT_CAM_FILE = str(
    Path(
        DATA_FOLDER,
        "composite_georef",
        "composite_georef_cameras.xml",
    )
)
# Example tree crown deliniation
DEFAULT_GEOPOLYGON_FILE = str(
    Path(DATA_FOLDER, "composite_georef", "composite_20230520T0519_tree_mask.geojson")
)
# Example field reference data
DEFAULT_GEO_POINTS_FILE = str(
    Path(DATA_FOLDER, "ept_trees_01_rectified_inclSmall.geojson")
)
# Example digital elevation model
DEFAULT_DTM_FILE = "/ofo-share/repos-david/semantic-mesh-pytorch3d/data/composite_georef/composite_georef_low_res_20230928T1637_dtm.tif"

## Misc constants
# Colors to be used across the project
COLORS = {
    "canopy": [34, 139, 34],
    "earth": [175, 128, 79],
}

VERT_ID = "vert_ID"
NULL_TEXTURE_FLOAT_VALUE = -1
NULL_TEXTURE_INT_VALUE = 255
