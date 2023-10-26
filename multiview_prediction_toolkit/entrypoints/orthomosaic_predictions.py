import argparse

from multiview_prediction_toolkit.segmentation import OrthoSegmentor
from pathlib import Path


def parse_args():
    """Parse and return arguements

    Returns:
        argparse.Namespace: Arguments
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--raster-image-file")
    parser.add_argument("--vector-label-file")
    parser.add_argument("--raster-label-file")

    parser.add_argument("--chip-size", type=int, default=2048)
    parser.add_argument("--training-stride", type=int, default=2048)
    parser.add_argument("--inference-stride", type=int, default=1024)

    parser.add_argument("--brightness-multiplier", type=float, default=1.0)

    parser.add_argument("--training-chips-folder", type=Path)
    parser.add_argument("--inference-chips-folder", type=Path)
    parser.add_argument("--prediction-chips-folder", type=Path)

    parser.add_argument("--aggregated-savefile", type=Path)
    parser.add_argument("--discard-edge-frac", type=float, default=0.125)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    ortho_seg = OrthoSegmentor(
        raster_input_file=args.raster_image_file,
        vector_label_file=args.vector_label_file,
        raster_label_file=args.raster_label_file,
        chip_size=args.chip_size,
        training_stride=args.training_stride,
        inference_stride=args.inference_stride,
    )

    if args.training_chips_folder is not None:
        ortho_seg.write_training_chips(
            args.training_chips_folder, brightness_multiplier=args.brightness_multiplier
        )

    if args.inference_chips_folder is not None:
        ortho_seg.write_inference_chips(
            args.inference_chips_folder,
            brightness_multiplier=args.brightness_multiplier,
        )

    if args.prediction_chips_folder is not None:
        ortho_seg.assemble_tiled_predictions(
            args.prediction_chips_folder,
            savefile=args.aggregated_savefile,
            discard_edge_frac=args.discard_edge_frac,
            eval_performance=True,
        )
