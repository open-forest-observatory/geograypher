import argparse
import json
from pathlib import Path

from geograypher.predictors.ortho_segmentor import write_chips


def parse_args():
    """Parse and return arguements

    Returns:
        argparse.Namespace: Arguments
    """

    description = (
        "This entrypoint is used for chipping raster data and optionally paired vector data. "
        + "All the arguments are passed to geograypher.entrypoints.write_chips "
        + "which has the following documentation:\n\n"
        + write_chips.__doc__
    )
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description=description
    )

    parser.add_argument("raster_file", type=Path)
    parser.add_argument("output_folder", type=Path)

    parser.add_argument("chip_size", type=int)
    parser.add_argument("chip_stride", type=int)

    parser.add_argument(
        "--label-vector-file", help="Path to vector label file.", type=Path
    )
    parser.add_argument("--label-column", type=str)
    parser.add_argument(
        "--label-remap",
        type=str,
        help="Provide this argument as a json-formatted string. "
        + 'For example \'{"name 1": 0, "name 2": 1}\'',
    )
    parser.add_argument("--write-empty-tiles", action="store_true")

    args = parser.parse_args()
    # Convert the json string representation to a dict
    args.label_remap = json.loads(args.label_remap)
    return args


if __name__ == "__main__":
    # Parse the arguments
    args = parse_args()
    # Pass them to the function
    write_chips(**args.__dict__)
