import numpy as np
from imageio import imread

from geograypher.constants import PATH_TYPE


def read_image_or_numpy(filename: PATH_TYPE) -> np.ndarray:
    """Read in a file that's either an image or numpy array

    Args:
        filename (PATH_TYPE): Filename to be read

    Raises:
        ValueError: If file cannot be read

    Returns:
        np.ndarray: Image, in format present on disk
    """
    img = None
    if img is None:
        try:
            img = imread(filename)
        except:
            print("Couldn't read as image")

    if img is None:
        try:
            img = np.load(filename)
        except:
            print("couldn't read as numpy")

    if img is None:
        raise ValueError("Could not read image")

    return img
