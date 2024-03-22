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
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--mesh-file",
        default=EXAMPLE_MESH_FILENAME,
        help="Path to the Metashape-exported mesh file, with associated transform .csv",
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
        "--label-folder",
        default=EXAMPLE_PREDICTED_LABELS_FOLDER,
        help="Path to the folder of labels to be aggregated onto the mesh. Must be in the same structure as the images.",
    )
    parser.add_argument(
        "--subset-images-folder", help="Use only images from this subset"
    )
    parser.add_argument(
        "--mesh-transform-file",
        help="Transform from the mesh coordinates to the earth-centered, earth-fixed frame.",
    )
    parser.add_argument(
        "--DTM-file",
        default=None,
        help="Optional path to a digital terrain model file to remove ground points",
    )
    parser.add_argument(
        "--height-above-ground-threshold",
        type=float,
        default=2,
        help="Height in meters above the DTM to consider ground. Only used if --DTM-file is set",
    )
    parser.add_argument("--ROI", help="Geofile region of interest to crop the mesh to")
    parser.add_argument(
        "--ROI-buffer-radius-meters",
        default=50,
        type=float,
        help="Keep points within this distance of the provided ROI object, if unset, everything will be kept",
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
        help="Downsample the mesh to this fraction of vertices for increased performance but lower quality",
    )
    parser.add_argument(
        "--aggregate-image-scale",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--aggregated-face-values-savefile",
        type=Path,
        help="Where to save the aggregated image values as a numpy array",
    )
    parser.add_argument(
        "--predicted-face-classes-savefile",
        type=Path,
        help="Where to save the most common label per face texture as a numpy array",
    )
    parser.add_argument(
        "--top-down-vector-projection-savefile",
        default="vis/predicted_map.geojson",
        help="Where to export the predicted map",
    )
    parser.add_argument("--vis", action="store_true", help="Show aggregated result")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # Parse command line args
    args = parse_args()
    # Pass command line args to aggregate_images
    aggregate_images(**args.__dict__)
