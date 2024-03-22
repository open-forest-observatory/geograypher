import argparse
import logging
from pathlib import Path

from geograypher.constants import (
    EXAMPLE_CAMERAS_FILENAME,
    EXAMPLE_IMAGE_FOLDER,
    EXAMPLE_MESH_FILENAME,
    EXAMPLE_RENDERED_LABELS_FOLDER,
    EXAMPLE_STANDARDIZED_LABELS_FILENAME,
)
from geograypher.entrypoints.workflow_functions import render_labels


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
        help="Path to the Metashape-exported mesh file",
    )
    parser.add_argument(
        "--cameras-file",
        default=EXAMPLE_CAMERAS_FILENAME,
        help="Path to the MetaShape-exported .xml camera file",
    )
    parser.add_argument(
        "--image-folder",
        default=EXAMPLE_IMAGE_FOLDER,
        help="Path to the folder of images used to create the mesh",
    )
    parser.add_argument(
        "--texture",
        default=EXAMPLE_STANDARDIZED_LABELS_FILENAME,
        help="File to load texture information from. "
        + "Must be a vector file open-able by geopandas, "
        + "a raster file open-able by rasterio "
        + "or a numpy file with the same number of elements as faces or vertics",
    )
    parser.add_argument(
        "--render-savefolder",
        default=EXAMPLE_RENDERED_LABELS_FOLDER,
        help="Where to render the labels",
    )
    parser.add_argument(
        "--subset-images-savefolder",
        help="Where to save the subset of images for which labels are generated",
        type=Path,
    )
    parser.add_argument(
        "--texture-column-name",
        default="Species",
        help="Column to use in vector file for texture information",
    )
    parser.add_argument(
        "--DTM-file",
        help="Path to a DTM file to use for ground thresholding",
    )
    parser.add_argument(
        "--ground-height-threshold",
        type=float,
        default=2.0,
        help="Set points under this height to ground. Only applicable if --DTM-file is set",
    )
    parser.add_argument(
        "--render-ground-class",
        action="store_true",
        help="Should the ground class be included in the renders or deleted.",
    )
    parser.add_argument(
        "--textured-mesh-savefile",
        help="Where to save the textured and subsetted mesh, if needed in the future",
    )
    parser.add_argument(
        "--ROI",
        help="The region of interest to render labels for",
    )
    parser.add_argument(
        "--ROI_buffer_radius_meters",
        default=50,
        type=float,
        help="The distance in meters to include around the ROI",
    )
    parser.add_argument(
        "--render-image-scale",
        type=float,
        default=0.25,
        help="Downsample the images to this fraction of the size for increased performance but lower quality",
    )
    parser.add_argument(
        "--mesh-downsample",
        type=float,
        default=1,
        help="Downsample the mesh to this fraction of vertices for increased performance but lower quality",
    )
    parser.add_argument(
        "--vis", action="store_true", help="Show mesh and rendered labels"
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    render_labels(**args.__dict__)
