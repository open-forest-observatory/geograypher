from multiview_prediction_toolkit.cameras import MetashapeCameraSet
from multiview_prediction_toolkit.config import (
    DEFAULT_CAM_FILE,
    DEFAULT_DEM_FILE,
    DEFAULT_GEOPOLYGON_FILE,
    DEFAULT_IMAGES_FOLDER,
    DEFAULT_LOCAL_MESH,
)
from multiview_prediction_toolkit.meshes import GeodataPhotogrammetryMesh

IMAGE_SCALE = 0.25

camera_set = MetashapeCameraSet(
    camera_file=DEFAULT_CAM_FILE, image_folder=DEFAULT_IMAGES_FOLDER
)
CAMERA_INDICES = list(range(0, camera_set.n_cameras(), 100))

mesh_w_colorspill = GeodataPhotogrammetryMesh(
    mesh_filename=DEFAULT_LOCAL_MESH, geo_polygon_file=DEFAULT_GEOPOLYGON_FILE
)
mesh_wout_colorspill = GeodataPhotogrammetryMesh(
    mesh_filename=DEFAULT_LOCAL_MESH,
    geo_polygon_file=DEFAULT_GEOPOLYGON_FILE,
    DEM_file=DEFAULT_DEM_FILE,
    ground_height_threshold=2,
)

mesh_w_colorspill.vis(
    interactive=True,
    screenshot_filename="vis/colorspill_mesh.png",
    window_size=(2000, 1600),
    title="colorspill",
)
mesh_wout_colorspill.vis(
    interactive=True,
    screenshot_filename="vis/no_colorspill_mesh.png",
    window_size=(2000, 1600),
    title="without colorspill",
)

mesh_w_colorspill.render_pytorch3d(
    camera_set=camera_set,
    image_scale=IMAGE_SCALE,
    camera_indices=CAMERA_INDICES,
    render_folder="colorspill",
)
mesh_wout_colorspill.render_pytorch3d(
    camera_set=camera_set,
    image_scale=IMAGE_SCALE,
    camera_indices=CAMERA_INDICES,
    render_folder="no_colorspill",
)
