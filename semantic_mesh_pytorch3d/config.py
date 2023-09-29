from pathlib import Path
from typing import Union

## Typing constants
# A file/folder path
PATH_TYPE = Union[str, Path]

## Path constants
# Where all the data is stored
DATA_FOLDER = Path(Path(__file__).parent, "..", "data").resolve()
# Images for one metashape project used as an example
DEFAULT_IMAGES_FOLDER = str(Path(DATA_FOLDER, "composite_georef", "images"))

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
    Path(DATA_FOLDER, "composite_georef", "composite_20230520T0519_crowns.gpkg")
)
# Example digital elevation model
DEFAULT_DEM = "/ofo-share/repos-david/semantic-mesh-pytorch3d/data/composite_georef/composite_georef_low_res_20230928T1637_dtm.tif"

## Misc constants
# Colors to be used across the project
COLORS = {
    "canopy": [34, 139, 34],
    "earth": [175, 128, 79],
}
