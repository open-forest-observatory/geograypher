from multiview_prediction_toolkit.meshes import GeodataPhotogrammetryMesh
from multiview_prediction_toolkit.config import (
    DEFAULT_GEOPOLYGON_FILE,
    DEFAULT_LOCAL_MESH,
    DEFAULT_GEO_POINTS_FILE,
)

mesh = GeodataPhotogrammetryMesh(
    DEFAULT_LOCAL_MESH, geo_point_file=DEFAULT_GEO_POINTS_FILE
)
# mesh = GeodataPhotogrammetryMesh(DEFAULT_LOCAL_MESH, geo_polygon_file=DEFAULT_GEOPOLYGON_FILE)
mesh.vis(interactive=True)
