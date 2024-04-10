from pathlib import Path

from geograypher.constants import PATH_TYPE


def ensure_folder(folder: PATH_TYPE):
    """Ensure this folder, and parent folders, exist. Nothing happens if already present

    Args:
        folder (PATH_TYPE): Path to folder to ensure exists
    """
    folder = Path(folder)
    # Create this folder and all parent folders if needed. Nothing happens if it already exists
    folder.mkdir(parents=True, exist_ok=True)


def ensure_containing_folder(filename: PATH_TYPE):
    """Ensure the folder containing this file exists. Nothing happens if already present.

    Args:
        filename (PATH_TYPE): The path to the file for which the containing folder should be created
    """
    # Cast the file to a pathlib Path
    filename = Path(filename)
    # Get the folder above it
    containing_folder = filename.parent
    # Create this folder and all parent folders if needed. Nothing happens if it already exists
    ensure_folder(containing_folder)
