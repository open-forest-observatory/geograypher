from pathlib import Path

DATA_FOLDER = Path(Path(__file__).parent, "..", "data").resolve()

DEFAULT_IMAGES_FOLDER = str(Path(DATA_FOLDER, "emerald_point_oblique_subset", "images"))

DEFAULT_LOCAL_MESH = str(
    Path(
        DATA_FOLDER,
        "emerald_point_oblique_subset",
        "exports",
        "emerald_point_oblique_subset_20230925T1512_model_local.ply",
    )
)

DEFAULT_GEOREF_MESH = str(
    Path(
        DATA_FOLDER,
        "emerald_point_oblique_subset",
        "exports",
        "emerald_point_oblique_subset_20230925T1512_model_georeferenced.ply",
    )
)

DEFAULT_CAM_FILE = str(
    Path(
        DATA_FOLDER,
        "emerald_point_oblique_subset",
        "exports",
        "emerald_point_oblique_subset_20230925T1512_cameras.xml",
    )
)

DEFAULT_GEOFILE = str(
    Path(DATA_FOLDER, "composite_20230520T0519", "composite_20230520T0519_crowns.gpkg")
)
