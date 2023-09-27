from pathlib import Path

DATA_FOLDER = Path(Path(__file__).parent, "..", "data").resolve()

DEFAULT_IMAGES_FOLDER = str(Path(DATA_FOLDER, "composite_20230520T0519", "images"))

DEFAULT_LOCAL_MESH = str(
    Path(
        DATA_FOLDER,
        "composite_georef",
        "composite_local_reset.ply",
    )
)

DEFAULT_CAM_FILE = str(
    Path(
        DATA_FOLDER,
        "composite_georef",
        "composite_georef_cameras.xml",
    )
)

DEFAULT_GEOPOLYGON_FILE = str(
    Path(DATA_FOLDER, "composite_20230520T0519", "composite_20230520T0519_crowns.gpkg")
)

COLORS = {
    "canopy": [[34, 139, 34]],
    "earth": [[175, 128, 79]],
}
