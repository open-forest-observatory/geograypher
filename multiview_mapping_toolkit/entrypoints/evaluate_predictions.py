import argparse
import json
from pathlib import Path

from multiview_mapping_toolkit.utils.prediction_metrics import (
    compute_rastervision_evaluation_metrics,
)


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--raster-file", type=Path, required=True, help="Path to image raster file"
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
    parser.add_argument(
        "--metrics-output-file",
        type=Path,
        help="Write out metrics to this file in json format",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    evaluation = compute_rastervision_evaluation_metrics(
        args.raster_file, args.prediction_file, args.groundtruth_file, args.class_names
    )
    eval_json = evaluation.to_json()
    if args.metrics_output_file is not None:
        with open(args.metrics_output_file, "w") as outfile_h:
            json.dump(eval_json, outfile_h)
    else:
        print(eval_json)
    breakpoint()
