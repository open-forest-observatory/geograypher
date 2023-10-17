from multiview_prediction_toolkit.cameras.derived_cameras import MetashapeCameraSet
from multiview_prediction_toolkit.meshes import (
    GeodataPhotogrammetryMesh,
)
from multiview_prediction_toolkit.config import (
    DATA_FOLDER,
    DEFAULT_CAM_FILE,
    DEFAULT_DEM_FILE,
    DEFAULT_IMAGES_FOLDER,
    DEFAULT_LOCAL_MESH,
    DEFAULT_GEO_POINTS_FILE,
)
import numpy as np
from pathlib import Path
from tqdm import tqdm
import os

from multiview_prediction_toolkit.segmentation.derived_segmentors import LookUpSegmentor
from multiview_prediction_toolkit.segmentation.segmentor import (
    SegmentorPhotogrammetryCameraSet,
)

IMAGE_SCALE = 0.25
IMAGES_FOLDER = Path(DATA_FOLDER, "training", "images")
RENDERS_FOLDER = Path(DATA_FOLDER, "training", "labels")
LABELS_FOLDER = Path(DATA_FOLDER, "composite_georef", "segmented")
SPECIES_PER_FACE_FILE = Path(DATA_FOLDER, "predicted_species_per_face.npy")

RELOAD_PRED_SPECIES = True
CREATE_LABELS = False

IMAGES_FOLDER.mkdir(exist_ok=True, parents=True)
RENDERS_FOLDER.mkdir(exist_ok=True)

if CREATE_LABELS:
    camera_set = MetashapeCameraSet(
        camera_file=DEFAULT_CAM_FILE, image_folder=DEFAULT_IMAGES_FOLDER
    )
    # Create mesh texture
    geodata_mesh = GeodataPhotogrammetryMesh(
        DEFAULT_LOCAL_MESH,
        geo_point_file=DEFAULT_GEO_POINTS_FILE,
    )
    geodata_mesh.vis(cmap="tab10")

    for i in tqdm(range(camera_set.n_cameras())):
        image = camera_set.get_image_by_index(i, image_scale=IMAGE_SCALE)
        image_path = camera_set.get_camera_by_index(i).image_filename
        label_mask = geodata_mesh.render_pytorch3d(
            camera_set, image_scale=IMAGE_SCALE, camera_index=i
        )
        np.save(Path(RENDERS_FOLDER, f"{i:06d}.npy"), label_mask)
        os.symlink(image_path, Path(IMAGES_FOLDER, f"{i:06d}{Path(image_path).suffix}"))
else:
    # Load information about a set of camera poses
    camera_set = MetashapeCameraSet(
        camera_file=DEFAULT_CAM_FILE, image_folder=DEFAULT_IMAGES_FOLDER
    )
    # Load the mesh and associated DEM file
    mesh = GeodataPhotogrammetryMesh(
        mesh_filename=DEFAULT_LOCAL_MESH,
        DEM_file=DEFAULT_DEM_FILE,
        downsample_target=0.25,
    )
    # Create a segmentor that looks up pre-processed images
    segmentor = LookUpSegmentor(
        base_folder=DEFAULT_IMAGES_FOLDER, lookup_folder=LABELS_FOLDER
    )
    # Make the camera set return segmented images instead of normal ones
    segmentor_camera_set = SegmentorPhotogrammetryCameraSet(
        camera_set, segmentor=segmentor
    )
    # Choose whether to skip this expensive step if it's already been computed
    if not RELOAD_PRED_SPECIES:
        # Find correspondences between image pixels and mesh faces
        species, _, _ = mesh.aggregate_viewpoints_pytorch3d(
            segmentor_camera_set, image_scale=IMAGE_SCALE
        )
        # Find the most common species for each face
        most_common_species_ID = np.argmax(species, axis=1)
        np.save(SPECIES_PER_FACE_FILE, most_common_species_ID)
    else:
        most_common_species_ID = np.load(SPECIES_PER_FACE_FILE).astype(float)

    # Set any points on the ground to not have a class
    is_ground = (mesh.get_height_above_ground(DEM_file=DEFAULT_DEM_FILE) < 2).astype(
        int
    )
    is_ground = mesh.vert_to_face_IDs(is_ground).astype(bool)
    most_common_species_ID[is_ground] = np.nan
    # Visualize
    mesh.export_face_labels_geofile(
        most_common_species_ID,
        "vis/predicted_species.geojson",
        label_names=(
            "ABCO",
            "ABMA",
            "CADE",
            "PI",
            "PICO",
            "PIJE",
            "PILA",
            "PIPO",
            "SLASCO",
            "TSME",
        ),
    )
    mesh.vis(vis_scalars=most_common_species_ID, cmap="tab10", interactive=True)
