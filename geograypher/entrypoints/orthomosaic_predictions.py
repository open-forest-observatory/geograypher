import argparse
import logging
from pathlib import Path

from geograypher.predictors.ortho_segmentor import (
    assemble_tiled_predictions,
    write_chips,
)


def parse_args():
    """Parse and return arguements

    Returns:
        argparse.Namespace: Arguments
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("raster_file", help="Path to raster tile")
    parser.add_argument("--vector-label-file", help="Path to vector label file.")
    parser.add_argument(
        "--vector-label-column",
        help="Which column in the vector data to use as the label. This column should be integer values. If unset the index will be used.",
    )
    parser.add_argument(
        "--write-empty-tiles",
        action="store_true",
        help="Write out training tiles that contain no labeled information",
    )

    parser.add_argument(
        "--chip-size", type=int, default=2048, help="Size of chips in pixels"
    )
    parser.add_argument(
        "--training-stride-fraction",
        type=int,
        default=0.5,
        help="The stride between chips as a fraction of the tile size",
    )
    parser.add_argument(
        "--inference-stride-fraction",
        type=int,
        default=0.5,
        help="The stride between chips as a fraction of the tile size",
    )

    parser.add_argument(
        "--training-chips-folder",
        type=Path,
        help="Run chipping for training and export to this folder",
    )
    parser.add_argument(
        "--inference-chips-folder",
        type=Path,
        help="Run chipping for inference and export to this folder",
    )
    parser.add_argument(
        "--prediction-chips-folder",
        type=Path,
        help="Run aggregation using chipts from this folder",
    )
    parser.add_argument(
        "--aggregated-savefile",
        type=Path,
        help="Save aggregated predictions to this file. Must be writable by rasterio. Only used with --prediction-chips-folder",
    )
    parser.add_argument(
        "--downweight-edge-frac",
        type=float,
        default=0.25,
        help="Downweight this fraction of predictions at the edges using a linear ramp. Only used with --prediction-chips-folder",
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

    if args.training_chips_folder is not None:
        logging.info(f"Writing training chips to {args.training_chips_folder}")
        write_chips(
            raster_file=args.raster_file,
            output_folder=args.training_chips_folder,
            chip_size=args.chip_size,
            chip_stride=int(args.training_stride_fraction * args.chip_size),
            label_vector_file=args.vector_label_file,
            label_column=args.vector_label_column,
            write_empty_tile=args.write_empty_tiles,
        )

    if args.inference_chips_folder is not None:
        logging.info(f"Writing inference chips to {args.inference_chips_folder}")
        write_chips(
            raster_file=args.raster_file,
            output_folder=args.inference_chips_folder,
            chip_size=args.chip_size,
            chip_stride=int(args.inference_stride_fraction * args.chip_size),
        )

    if args.prediction_chips_folder is not None:
        logging.info(
            f"Aggregating tiled predictions from {args.prediction_chips_folder}"
            + (
                ""
                if args.aggregated_savefile is None
                else f" and saving to {args.aggregated_savefile}"
            )
        )
        assemble_tiled_predictions(
            raster_input_file=args.raster_file,
            pred_files=sorted(args.prediction_chips_folder.glob("*")),
            class_savefile=args.aggregated_savefile,
            downweight_edge_frac=args.downweight_edge_frac,
        )
