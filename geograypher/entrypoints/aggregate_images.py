import argparse
from pathlib import Path

from geograypher.constants import (
    EXAMPLE_CAMERAS_FILENAME,
    EXAMPLE_IDS_TO_LABELS,
    EXAMPLE_IMAGE_FOLDER,
    EXAMPLE_MESH_FILENAME,
    EXAMPLE_PREDICTED_LABELS_FOLDER,
)
from geograypher.entrypoints.workflow_functions import aggregate_images


def parse_args():
    description = (
        "This script aggregates predictions from individual images onto the mesh. This aggregated "
        + "prediction can then be exported into geospatial coordinates. The default option is to "
        + "use the provided example data. All of the arguments are passed to "
        + "geograypher.entrypoints.workflow_functions.aggregate_images "
        + "which has the following documentation:\n\n"
        + aggregate_images.__doc__
    )
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description=description
    )
    parser.add_argument(
        "--mesh-file",
        default=EXAMPLE_MESH_FILENAME,
    )
    parser.add_argument(
        "--cameras-file",
        default=EXAMPLE_CAMERAS_FILENAME,
    )
    parser.add_argument(
        "--image-folder",
        default=EXAMPLE_IMAGE_FOLDER,
    )
    parser.add_argument(
        "--label-folder",
        default=EXAMPLE_PREDICTED_LABELS_FOLDER,
    )
    parser.add_argument(
        "--subset-images-folder",
    )
    parser.add_argument(
        "--mesh-transform-file",
    )
    parser.add_argument(
        "--DTM-file",
    )
    parser.add_argument(
        "--height-above-ground-threshold",
        type=float,
        default=2,
    )
    parser.add_argument("--ROI")
    parser.add_argument(
        "--ROI-buffer-radius-meters",
        default=50,
        type=float,
    )
    parser.add_argument(
        "--IDs-to-labels",
        default=EXAMPLE_IDS_TO_LABELS,
        type=dict,
    )
    parser.add_argument(
        "--mesh-downsample",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--aggregate-image-scale",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--aggregated-face-values-savefile",
        type=Path,
    )
    parser.add_argument(
        "--predicted-face-classes-savefile",
        type=Path,
    )
    parser.add_argument(
        "--top-down-vector-projection-savefile",
        default="vis/predicted_map.geojson",
    )
    parser.add_argument("--vis", action="store_true")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # Parse command line args
    args = parse_args()
    # Pass command line args to aggregate_images
    aggregate_images(**args.__dict__)
