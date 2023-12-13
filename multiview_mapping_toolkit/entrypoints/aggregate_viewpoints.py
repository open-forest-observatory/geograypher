import argparse
import logging
from pathlib import Path

import numpy as np

from multiview_mapping_toolkit.cameras import MetashapeCameraSet
from multiview_mapping_toolkit.config import (
    EXAMPLE_CAMERAS_FILENAME,
    EXAMPLE_IMAGE_FOLDER,
    EXAMPLE_MESH_FILENAME,
    EXAMPLE_PREDICTED_LABELS_FOLDER,
)
from multiview_mapping_toolkit.meshes import TexturedPhotogrammetryMesh
from multiview_mapping_toolkit.segmentation import (
    LookUpSegmentor,
    SegmentorPhotogrammetryCameraSet,
)


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--camera-file",
        default=EXAMPLE_CAMERAS_FILENAME,
        help="Path to the MetaShape-exported .xml camera file",
    )
    parser.add_argument(
        "--mesh-file",
        default=EXAMPLE_MESH_FILENAME,
        help="Path to the Metashape-exported mesh file, with associated transform .csv",
    )
    parser.add_argument(
        "--mesh-ROI", help="Geofile region of interest to crop the mesh to"
    )
    parser.add_argument(
        "--mesh-ROI-buffer-meters",
        type=float,
        help="Keep points within this distance of the provided ROI object, if unset, everything will be kept",
    )
    parser.add_argument(
        "--image-folder",
        default=EXAMPLE_IMAGE_FOLDER,
        help="Path to the folder of images used to create the mesh",
    )
    parser.add_argument(
        "--label-folder",
        default=EXAMPLE_PREDICTED_LABELS_FOLDER,
        help="Path to the folder of labels to be aggregated onto the mesh. Must be in the same structure as the images.",
    )
    parser.add_argument(
        "--DTM-file",
        default=None,
        help="Optional path to a digital terrain model file to remove ground points",
    )
    parser.add_argument(
        "--ground-height-threshold",
        type=float,
        default=2,
        help="Height in meters above the DTM to consider ground. Only used if --DTM-file is set",
    )
    parser.add_argument(
        "--export-file",
        default="vis/predicted_map.geojson",
        help="Where to export the predicted map",
    )
    parser.add_argument(
        "--texture-export-filename-npy",
        type=Path,
        help="Where to save the mesh texture as a numpy array",
    )
    parser.add_argument(
        "--mesh-downsample",
        type=float,
        default=0.25,
        help="Downsample the mesh to this fraction of vertices for increased performance but lower quality",
    )
    parser.add_argument(
        "--image-downsample",
        type=float,
        default=0.25,
        help="Downsample the images to this fraction of the size for increased performance but lower quality",
    )
    parser.add_argument(
        "--rendering-batch-size",
        type=int,
        default=1,
        help="The number of images to render at once",
    )
    parser.add_argument("--label-names", nargs="+", help="Optional of label names")
    parser.add_argument(
        "--num-classes",
        type=int,
        default=10,
        help="Number of classes in segmentation task",
    )
    parser.add_argument("--vis", action="store_true", help="Show aggregated result")
    parser.add_argument(
        "--log-level",
        default="info",
        choices=list(logging._nameToLevel.keys()),
        help="Verbosity of printouts",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    logging.basicConfig(level=args.log_level.upper())

    # Load information about a set of camera poses
    logging.info("Creating camera set")
    camera_set = MetashapeCameraSet(
        camera_file=args.camera_file, image_folder=args.image_folder
    )
    if args.mesh_ROI is not None and args.mesh_ROI_buffer_meters is not None:
        camera_set = camera_set.get_subset_ROI(
            ROI=args.mesh_ROI, buffer_radius_meters=args.mesh_ROI_buffer_meters
        )
        print(len(camera_set.cameras))

    # Load the mesh
    logging.info("Creating mesh")
    mesh = TexturedPhotogrammetryMesh(
        mesh=args.mesh_file,
        downsample_target=args.mesh_downsample,
        transform_filename=args.camera_file,
        ROI=args.mesh_ROI,
        ROI_buffer_meters=args.mesh_ROI_buffer_meters,
    )

    # Create a segmentor that looks up pre-processed images
    logging.info("Creating lookup segmentor")
    # Set number of classes if only names is provided
    if args.num_classes is None and args.class_names is not None:
        args.num_classes = len(args.class_names)

    # Create the image segmentor that reads from predictions on disk
    segmentor = LookUpSegmentor(
        base_folder=args.image_folder,
        lookup_folder=args.label_folder,
        num_classes=args.num_classes,
    )
    # Make the camera set return segmented images instead of normal ones
    logging.info("Creating segmentor camera set")
    segmentor_camera_set = SegmentorPhotogrammetryCameraSet(
        camera_set, segmentor=segmentor
    )

    # Choose whether to skip this expensive step if it's already been computed
    # Find correspondences between image pixels and mesh faces
    logging.info("Main step: aggregating viewpoints")
    averaged_label_IDs, _, _ = mesh.aggregate_viewpoints_pytorch3d(
        segmentor_camera_set,
        image_scale=args.image_downsample,
        batch_size=args.rendering_batch_size,
    )
    # Find the most common species for each face
    most_common_label_ID = np.argmax(averaged_label_IDs, axis=1)

    if args.DTM_file is not None:
        logging.info("Thresholding based on height above DTM")
        # Set any points on the ground to not have a class
        most_common_label_ID = mesh.label_ground_class(
            DTM_file=args.DTM_file,
            height_above_ground_threshold=args.ground_height_threshold,
            label=most_common_label_ID,
            ground_ID=np.nan,
        )

    # Export the predictions as a numpy file
    if args.texture_export_filename_npy is not None:
        logging.info("Exporting predictions to numpy file")
        args.texture_export_filename_npy.parent.mkdir(exist_ok=True, parents=True)
        np.save(args.texture_export_filename_npy, most_common_label_ID)

    # Export the predictions
    logging.info("Exporting predictions to vector file")
    mesh.export_face_labels_vector(
        face_labels=most_common_label_ID,
        export_file=args.export_file,
        label_names=args.label_names,
    )
    # Visualize
    if args.vis:
        logging.info("Visualizing result")
        mesh.vis(
            vis_scalars=most_common_label_ID,
            interactive=True,
            mesh_kwargs={"cmap": "tab10", "clim": [0, 9]},
        )
