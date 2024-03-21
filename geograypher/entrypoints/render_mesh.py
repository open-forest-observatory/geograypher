import argparse
import logging
from pathlib import Path

from geograypher.cameras import MetashapeCameraSet
from geograypher.constants import (
    EXAMPLE_CAMERAS_FILENAME,
    EXAMPLE_DTM_FILE,
    EXAMPLE_IMAGE_FOLDER,
    EXAMPLE_MESH_FILENAME,
    EXAMPLE_RENDERED_LABELS_FOLDER,
    EXAMPLE_STANDARDIZED_LABELS_FILENAME,
    TEN_CLASS_VIS_KWARGS,
)
from geograypher.meshes import TexturedPhotogrammetryMesh


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
        "--DTM-file",
        help="Path to a DTM file to use for ground thresholding",
    )
    parser.add_argument(
        "--vector-file",
        default=EXAMPLE_STANDARDIZED_LABELS_FILENAME,
        help="Vector file to load texture information from. Must be open-able by geopandas",
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
    parser.add_argument(
        "--vector-file-column",
        default="Species",
        help="Column to use in vector file for texture information",
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
        "--ground-height-threshold",
        type=float,
        default=2.0,
        help="Set points under this height to ground. Only applicable if --DTM-file is set",
    )
    parser.add_argument(
        "--ROI-buffer-meters",
        type=float,
        help="Remove all portions of the mesh that are farther than this distance in meters"
        + " from the labeled data. If unset, the entire mesh will be retained.",
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

    # Setup logging
    logging.basicConfig(level=args.log_level.upper())
    logger = logging.getLogger(__name__)

    # Load the camera set
    logger.info("Creating the camera set")
    camera_set = MetashapeCameraSet(args.camera_file, args.image_folder)
    if args.ROI_buffer_meters is not None:
        logger.info("Subsetting cameras")
        camera_set = camera_set.get_subset_ROI(
            ROI=args.vector_file, buffer_radius_meters=args.ROI_buffer_meters
        )
        if args.save_subset_images_folder:
            logger.info("Saving subset of images")
            camera_set.save_images(args.save_subset_images_folder)

    # Load the mesh
    logger.info("Loading the mesh")
    mesh = TexturedPhotogrammetryMesh(
        args.mesh_file,
        downsample_target=args.mesh_downsample,
        transform_filename=args.camera_file,
        texture=args.vector_file,
        texture_column_name=args.vector_file_column,
        ROI=args.vector_file if args.ROI_buffer_meters is not None else None,
        ROI_buffer_meters=args.ROI_buffer_meters,
        require_transform=True,
    )

    if args.DTM_file is not None:
        # Load the mesh
        logger.info("Setting the ground class based on the DTM")
        mesh.label_ground_class(
            DTM_file=args.DTM_file,
            height_above_ground_threshold=args.ground_height_threshold,
            set_mesh_texture=True,
        )

    if args.vis or args.screenshot_filename is not None:
        logger.info("Visualizing the mesh")
        mesh.vis(
            screenshot_filename=args.screenshot_filename,
        )

    logger.info("Rendering the images")
    mesh.save_renders_pytorch3d(
        camera_set=camera_set,
        render_image_scale=args.image_downsample,
        output_folder=args.render_folder,
        make_composites=False,
        save_native_resolution=True,
    )
