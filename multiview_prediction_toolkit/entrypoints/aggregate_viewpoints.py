import argparse
import numpy as np
from multiview_prediction_toolkit.meshes import TexturedPhotogrammetryMesh
from multiview_prediction_toolkit.segmentation import (
    SegmentorPhotogrammetryCameraSet,
    LookUpSegmentor,
)
from multiview_prediction_toolkit.cameras import MetashapeCameraSet

from multiview_prediction_toolkit.config import (
    DEFAULT_IMAGES_FOLDER,
    DEFAULT_LOCAL_MESH,
    DEFAULT_DEM_FILE,
    DEFAULT_CAM_FILE,
    DEFAULT_LABELS_FOLDER,
)

IMAGE_SCALE = 0.25


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera-file", default=DEFAULT_CAM_FILE)
    parser.add_argument("--mesh-file", default=DEFAULT_LOCAL_MESH)
    parser.add_argument("--image-folder", default=DEFAULT_IMAGES_FOLDER)
    parser.add_argument("--label-folder", default=DEFAULT_LABELS_FOLDER)
    parser.add_argument("--DEM-file", default=DEFAULT_DEM_FILE)
    parser.add_argument("--export-file", default="vis/predicted_map.geojson")
    parser.add_argument("--mesh-downsample", type=float, default=0.25)
    parser.add_argument("--image-downsample", type=float, default=0.25)
    parser.add_argument("--ground-height-threshold", type=float, default=2)
    parser.add_argument("--label-names", nargs="+")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    # Load information about a set of camera poses
    camera_set = MetashapeCameraSet(
        camera_file=args.camera_file, image_folder=args.image_folder
    )
    # Load the mesh and associated DEM file
    mesh = TexturedPhotogrammetryMesh(
        mesh_filename=args.mesh_file,
        downsample_target=args.mesh_downsample,
    )
    # Create a segmentor that looks up pre-processed images
    segmentor = LookUpSegmentor(
        base_folder=args.image_folder, lookup_folder=args.label_folder
    )
    # Make the camera set return segmented images instead of normal ones
    segmentor_camera_set = SegmentorPhotogrammetryCameraSet(
        camera_set, segmentor=segmentor
    )

    # Set any points on the ground to not have a class
    is_ground = mesh.get_height_above_ground(
        DEM_file=args.DEM_file, threshold=args.ground_height_threshold
    ).astype(int)

    # Choose whether to skip this expensive step if it's already been computed
    # Find correspondences between image pixels and mesh faces
    species, _, _ = mesh.aggregate_viewpoints_pytorch3d(
        segmentor_camera_set, image_scale=args.image_downsample
    )
    # Find the most common species for each face
    most_common_species_ID = np.argmax(species, axis=1)
    is_ground = mesh.vert_to_face_IDs(is_ground).astype(bool)
    most_common_species_ID[is_ground] = np.nan

    # Export the predictions
    mesh.export_face_labels_geofile(
        most_common_species_ID,
        args.export_file,
        label_names=args.label_names,
    )
    # Visualize
    mesh.vis(
        vis_scalars=most_common_species_ID,
        interactive=True,
        mesh_kwargs={"cmap": "tab10", "clim": [0, 9]},
    )
