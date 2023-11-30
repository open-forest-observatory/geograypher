import argparse
import logging
from pathlib import Path

import numpy as np
from tqdm import tqdm

from multiview_prediction_toolkit.cameras import MetashapeCameraSet
from multiview_prediction_toolkit.config import (
    DATA_FOLDER,
    EXAMPLE_CAMERAS_FILENAME,
    EXAMPLE_STANDARDIZED_LABELS_FILENAME,
    EXAMPLE_IMAGE_FOLDER,
    EXAMPLE_MESH_FILENAME,
)
from multiview_prediction_toolkit.meshes import TexturedPhotogrammetryMesh


def parse_args():
    """Parse and return arguements

    Returns:
        argparse.Namespace: Arguments
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--mesh-file",
        default=EXAMPLE_MESH_FILENAME,
        help="Path to the Metashape-exported mesh file, with associated transform .csv",
    )
    parser.add_argument(
        "--camera-file",
        default=EXAMPLE_CAMERAS_FILENAME,
        help="Path to the MetaShape-exported .xml camera file",
    )
    parser.add_argument(
        "--image-folder",
        default=EXAMPLE_IMAGE_FOLDER,
        help="Path to the folder of images used to create the mesh",
    )
    parser.add_argument(
        "--render-folder",
        default="vis/example_renders",
        help="Path to save the rendered images. Will be created if not present",
    )
    parser.add_argument(
        "--mesh-downsample",
        type=float,
        default=1,
        help="Downsample the mesh to this fraction of vertices for increased performance but lower quality",
    )
    parser.add_argument(
        "--image-downsample",
        type=float,
        default=0.25,
        help="Downsample the images to this fraction of the size for increased performance but lower quality",
    )
    parser.add_argument(
        "--vector-file",
        default=EXAMPLE_STANDARDIZED_LABELS_FILENAME,
        help="Vector file to load texture information from. Must be open-able by geopandas",
    )
    parser.add_argument(
        "--vector-file-column",
        default="Species",
        help="Column to use in vector file for texture information",
    )
    parser.add_argument(
        "--ROI-buffer-meters",
        type=float,
        help="Remove all portions of the mesh that are farther than this distance in meters"
        + " from the labeled data. If unset, the entire mesh will be retained.",
    )
    parser.add_argument(
        "--save-subset-images-folder",
        help="Where to save the subset of images near the labeled data",
        type=Path,
    )
    parser.add_argument(
        "--render-folder",
        default=EXAMPLE_RENDERED_LABELS_FOLDER,
        help="Where to render the labels",
    )
    parser.add_argument("--vis", action="store_true", help="Show mesh")
    parser.add_argument(
        "--screenshot-filename", help="If provided, save mesh render to this file"
    )
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

    # Load the camera set
    logging.info("Creating the camera set")
    camera_set = MetashapeCameraSet(args.camera_file, args.image_folder)
    if args.ROI_buffer_meters is not None:
        logging.info("Subsetting cameras")
        camera_set = camera_set.get_subset_near_geofile(
            args.vector_file, args.ROI_buffer_meters
        )
        if args.save_subset_images_folder:
            logging.info("Saving subset of images")
            camera_set.save_images(args.save_subset_images_folder)

    # Load the mesh
    logging.info("Loading the mesh")
    mesh = TexturedPhotogrammetryMesh(
        args.mesh_file, downsample_target=args.mesh_downsample
    )

    logging.info("Setting the mesh texture")
    mesh.get_values_for_verts_from_vector(
        column_names=args.vector_file_column,
        vector_file=args.vector_file,
        set_vertex_IDs=True,
    )

    if args.vis or args.screenshot_filename is not None:
        logging.info("Visualizing the mesh")
        mesh.vis(screenshot_filename=args.screenshot_filename, camera_set=camera_set)

    mesh.save_renders_pytorch3d(
        camera_set=camera_set,
        render_image_scale=args.image_downsample,
        output_folder=args.render_folder,
        make_composite=False,
    )
