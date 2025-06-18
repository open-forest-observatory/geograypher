import argparse
from inspect import signature
from pathlib import Path

from geograypher.predictors.ortho_segmentor import assemble_tiled_predictions


def parse_args():
    """Parse and return arguements

    Returns:
        argparse.Namespace: Arguments
    """

    # For optional CLI arguments we must provide a default value. In most cases, these parameters
    # already have a default defined at the function level.
    atp_params = signature(assemble_tiled_predictions).parameters
    # Build the description string
    description = (
        "This entrypoint is used for aggregating predictions on tiled data into a single geospatial "
        + "raster. "
        + "All the arguments are passed to geograypher.entrypoints.assemble_tiled_predictions "
        + "which has the following documentation:\n\n"
        + assemble_tiled_predictions.__doc__
    )

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description=description
    )
    parser.add_argument("raster_file", type=Path)
    parser.add_argument("pred_folder", type=Path)
    parser.add_argument("class_savefile", type=Path)

    parser.add_argument("num_classes", type=int)
    parser.add_argument("--counts-savefile", type=Path)
    parser.add_argument(
        "--downweight-edge-frac",
        type=float,
        default=atp_params["downweight_edge_frac"].default,
    )
    parser.add_argument(
        "--nodataval",
        type=int,
        default=atp_params["nodataval"].default,
    )
    # TODO consider adding the argument for count_dtype
    # the challenge here is we need to parse a class type (ex. np.uint8, np.uint16)
    parser.add_argument(
        "--max-overlapping-tiles",
        type=int,
        default=atp_params["max_overlapping_tiles"].default,
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # Parse the args
    args = parse_args()
    # Unpack them to the function
    assemble_tiled_predictions(**args.__dict__)
