import argparse
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
    description = (
        "This script renders labels onto individual images using geospatial textures. "
        + "By default is uses the example data. All arguments are passed to "
        + "geograypher.entrypoints.workflow_functions.render_labels "
        + "which has the following documentation:\n\n"
        + render_labels.__doc__
    )
    # Ideally we'd include the defaults for each argument, but there is no help text so the
    # ArgumentDefaultsHelpFormatter formatter doesn't show them
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Add arguments
    parser.add_argument("--mesh-file", default=EXAMPLE_MESH_FILENAME)
    parser.add_argument(
        "--cameras-file",
        default=EXAMPLE_CAMERAS_FILENAME,
    )
    parser.add_argument(
        "--image-folder",
        default=EXAMPLE_IMAGE_FOLDER,
    )
    parser.add_argument(
        "--texture",
        default=EXAMPLE_STANDARDIZED_LABELS_FILENAME,
    )
    parser.add_argument(
        "--render-savefolder",
        default=EXAMPLE_RENDERED_LABELS_FOLDER,
    )
    parser.add_argument(
        "--subset-images-savefolder",
        type=Path,
    )
    parser.add_argument(
        "--texture-column-name",
        default="Species",
    )
    parser.add_argument(
        "--DTM-file",
    )
    parser.add_argument(
        "--ground-height-threshold",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--render-ground-class",
        action="store_true",
    )
    parser.add_argument(
        "--textured-mesh-savefile",
    )
    parser.add_argument(
        "--ROI",
    )
    parser.add_argument(
        "--ROI_buffer_radius_meters",
        default=50,
        type=float,
    )
    parser.add_argument(
        "--render-image-scale",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--mesh-downsample",
        type=float,
        default=1,
    )
    parser.add_argument("--vis", action="store_true")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # Parse command line args
    args = parse_args()
    # Pass all the arguments command line options to render_labels
    render_labels(**args.__dict__)
