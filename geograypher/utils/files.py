from pathlib import Path

from geograypher.constants import PATH_TYPE


def ensure_containing_folder(filename: PATH_TYPE):
    # Cast the file to a pathlib Path
    filename = Path(filename)
    # Get the folder above it
    containing_folder = filename.parent
    # Create this folder and all parent folders if needed. Nothing happens if it already exists
    containing_folder.mkdir(parents=True, exist_ok=True)
