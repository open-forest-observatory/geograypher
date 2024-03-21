import argparse
import json
from pathlib import Path

from geograypher.utils.prediction_metrics import (
    compute_confusion_matrix_from_geospatial,
)


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--prediction-file",
        type=Path,
        required=True,
        help="Path to vector or raster prediction file",
    )
    parser.add_argument(
        "--groundtruth-file",
        type=Path,
        required=True,
        help="Path to vector or raster label file",
    )
    parser.add_argument("--class-names", nargs="+")
    parser.add_argument("--vis-savefile")
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--column-name")
    parser.add_argument(
        "--metrics-output-file",
        type=Path,
        help="Write out metrics to this file in json format",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    cf_matrix, classes, accuracy = compute_confusion_matrix_from_geospatial(
        prediction_file=args.prediction_file,
        groundtruth_file=args.groundtruth_file,
        class_names=args.class_names,
        vis_savefile=args.vis_savefile,
        normalize=args.normalize,
        column_name=args.column_name,
    )
