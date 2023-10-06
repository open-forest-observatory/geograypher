from multiview_prediction_toolkit.meshes import GeodataPhotogrammetryMesh
from multiview_prediction_toolkit.config import (
    DEFAULT_LOCAL_MESH,
    DEFAULT_GEO_POINTS_FILE,
)

GeodataPhotogrammetryMesh(DEFAULT_LOCAL_MESH, geo_polygon_file=DEFAULT_GEO_POINTS_FILE)
