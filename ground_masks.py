import numpy as np
import pyvista as pv
from pathlib import Path

from geograypher.cameras.derived_cameras import MetashapeCameraSet
from geograypher.meshes.meshes import TexturedPhotogrammetryMesh

IMAGE_FOLDER = "/ofo-share/scratch-eric/000936-01/00"
CAMERA_XML = "/ofo-share/scratch-eric/000936-01/000936_01_cameras.xml"
DTM_FILE = "/ofo-share/scratch-eric/000936-01/000936_01_dtm-ptcloud.tif"
OUTPUT_FOLDER = "/ofo-share/scratch-eric/000936-01/00-ground"
MESH_FILE = "/ofo-share/scratch-eric/000936-01/000936_01_model-local.ply"
# Optional much smaller mesh for speed reasons, choose one or the other
# MESH_FILE = "/ofo-share/scratch-eric/000936-01/000936_01_model-local_10PCT.ply"

mesh = TexturedPhotogrammetryMesh(
    MESH_FILE, transform_filename=CAMERA_XML, require_transform=True,
)

# Sort the ground height into texture, so that 0=nan, 1=ground, 2=aboveground
CUTOFF = 1.0
ground_mask = mesh.get_height_above_ground(DTM_file=DTM_FILE)
texture = np.zeros(len(ground_mask), dtype=float)
texture[ground_mask > CUTOFF] = 2
texture[ground_mask <= CUTOFF] = 1
texture[np.isnan(ground_mask)] = 0

camera_set = MetashapeCameraSet(CAMERA_XML, IMAGE_FOLDER)
ground_mesh = TexturedPhotogrammetryMesh(
    MESH_FILE,
    transform_filename=CAMERA_XML,
    require_transform=True,
    texture=texture.reshape(-1, 1),
)

ground_mesh.save_renders(
    camera_set,
    output_folder=OUTPUT_FOLDER,
    save_native_resolution=True,
)

