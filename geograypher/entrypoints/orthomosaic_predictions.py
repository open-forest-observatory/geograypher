import argparse
import logging
from pathlib import Path

from geograypher.predictors import OrthoSegmentor


def parse_args():
    """Parse and return arguements

    Returns:
        argparse.Namespace: Arguments
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("raster_image_file", help="Path to raster tile")
    parser.add_argument(
        "--vector-label-file",
        help="Path to vector label file. Cannot be used with --raster-label-file",
    )
    parser.add_argument(
        "--raster-label-file",
        help="Path to raster label file. Cannot be used with --vector-label-file",
    )

    parser.add_argument("--class-names", nargs="+")

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
        "--brightness-multiplier",
        type=float,
        default=1.0,
        help="Multiplier on chip brightness",
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
    parser.add_argument("--class-names", nargs="+")
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

    logging.info("Loading data for ortho segmentor class")
    ortho_seg = OrthoSegmentor(
        raster_input_file=args.raster_image_file,
        vector_label_file=args.vector_label_file,
        raster_label_file=args.raster_label_file,
        class_names=args.class_names,
        chip_size=args.chip_size,
        training_stride=int(args.training_stride_fraction * args.chip_size),
        inference_stride=int(args.inference_stride_fraction * args.chip_size),
        class_names=args.class_names,
    )

    if args.training_chips_folder is not None:
        logging.info(f"Writing training chips to {args.training_chips_folder}")
        ortho_seg.write_training_chips(
            args.training_chips_folder, brightness_multiplier=args.brightness_multiplier
        )

    if args.inference_chips_folder is not None:
        logging.info(f"Writing inference chips to {args.inference_chips_folder}")
        ortho_seg.write_inference_chips(
            args.inference_chips_folder,
            brightness_multiplier=args.brightness_multiplier,
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
        ortho_seg.assemble_tiled_predictions(
            sorted(args.prediction_chips_folder.glob("*")),
            class_savefile=args.aggregated_savefile,
            downweight_edge_frac=args.downweight_edge_frac,
        )
