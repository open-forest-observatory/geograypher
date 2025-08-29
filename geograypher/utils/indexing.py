import logging

import numpy as np
import pandas as pd
from scipy.interpolate import griddata


def find_argmax_nonzero_value(
    array: np.ndarray, keepdims: bool = False, axis: int = 1
) -> np.array:
    """Find the argmax of an array, setting entires with zero sum or finite values to nan

    Args:
        array (np.ndarray): The input array
        keepdims (bool, optional): Should the dimensions be kept. Defaults to False.
        axis (int, optional): Which axis to perform the argmax along. Defaults to 1.

    Returns:
        np.array: The argmax, with nans for invalid or all-zero entries
    """
    # Find the column with the highest value per row
    argmax = np.argmax(array, axis=axis, keepdims=keepdims).astype(float)

    # Find rows with zero sum or any infinite values
    zero_sum_mask = np.sum(array, axis=axis) == 0
    infinite_mask = np.any(~np.isfinite(array), axis=axis)

    # Set these rows in the argmax to nan
    argmax[np.logical_or(zero_sum_mask, infinite_mask)] = np.nan

    return argmax


def ensure_float_labels(query_array, full_array=None) -> (np.ndarray, dict):
    # Standardizing the type
    if isinstance(query_array, pd.Series):
        query_array = query_array.to_numpy()
    else:
        query_array = np.array(query_array)

    # Check if the array is a string or some other object type that is not numeric
    if query_array.dtype in (str, np.dtype("O")):
        # Get unique values from full array if provided, or query array if not
        # Note that if the full array is not provided, the IDs will change based
        # on what set of the labeled data is indexed
        unique_values = np.unique(query_array if full_array is None else full_array)

        # Build an output array
        output_query_array = np.full_like(query_array, fill_value=np.nan, dtype=float)

        IDs_to_label = {}
        # Iterate through and set the values that match to the integer label
        for i, unique_value in enumerate(unique_values):
            output_query_array[query_array == unique_value] = i
            IDs_to_label[i] = unique_value

        return output_query_array, IDs_to_label

    # This is numeric data, but we want to check if it's representing something
    # categorical/ordinal

    # Determine which labels are finite so we can cast them to ints
    finite_labels = query_array[np.isfinite(query_array)]
    # See if all labels are approximately ints
    if np.allclose(finite_labels, finite_labels.astype(int)):
        # Try to remove any numerical issues by rounding to the ints
        unique_values = np.unique(
            np.round(query_array[np.isfinite(query_array)])
            if full_array is None
            else np.round(full_array[np.isfinite(full_array)])
        )

        IDs_to_label = {i: val for i, val in enumerate(unique_values)}
    else:
        # These are not discrete, so it doesn't make sense to represent them with IDs
        IDs_to_label = None

    # Cast to float, since that's expected
    output_query_array = query_array.astype(float)
    return output_query_array, IDs_to_label


def inverse_map_interpolation(
    ijmap: np.ndarray, downsample: int = 1, fill: int = -1
) -> np.ndarray:
    """
    Inverts the type of map that can be fed to skimage.transform.warp.

    The basic construction is the pixel position of these maps is the position
    in the destination image, and the value of the map is the position in the
    source image. Therefore if location [20, 30] has value [22.2, 28.4], it
    means that the destination image pixel [20, 30] will be sampled from the
    source image at pixel [22.2, 28.4] (the sampler can choose to snap to the
    closest integer value or interpolate nearby pixels).

    By inverting, we seek to reverse this map through interpolation. In the
    example above, in the output we would now have the location [22, 28]
    have the value [20, 30] (or slightly different because it might
    interpolate with nearby values).

    Arguments:
        ijmap ((2, H, W) numpy array): a map of the structure discussed above
        downsample (int): Inverting (particularly griddata) can be expensive
            for high-res image maps. You can downsample in integer steps to
            save computation, which will downsample both the (i, j) axes. Using
            downsample=2 will therefore interpolate on 1/4 of the pixels.
        fill (int): Values outside of the interpolation convex hull will take
            this value.

    Returns:
        (2, H, W) numpy array of the same shape as ijmap
    """

    # (row, col) grid of pixels coordinates
    H, W = ijmap.shape[1:]
    igrid, jgrid = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")

    # Get an (N, 2) array of the grid coordinates
    grid_coords = np.stack([igrid.ravel(), jgrid.ravel()], axis=1)

    # Get (N, 2) arrays of the source and goal coordinates we have data for
    if downsample > 1:
        ds = slice(None, None, downsample)
        sample_y = np.stack([igrid[ds, ds].ravel(), jgrid[ds, ds].ravel()], axis=1)
        sample_x = np.stack(
            [ijmap[0][ds, ds].ravel(), ijmap[1][ds, ds].ravel()], axis=1
        )
    else:
        sample_y = grid_coords.copy()
        sample_x = np.stack([ijmap[0].ravel(), ijmap[1].ravel()], axis=1)

    # This is a little complicated, but griddata takes in
    # 1) The samples you have data for (x)
    # 2) The sample data (y)
    # 3) The new x at which you want to resample
    # In this case our sample x data is the mapped indices, the sample y data
    # is the grid that mapping came from, and the resample x is also the grid
    # because we are trying to invert.
    inv_i = griddata(
        sample_x, sample_y[:, 0], grid_coords, method="linear", fill_value=fill
    )
    inv_j = griddata(
        sample_x, sample_y[:, 1], grid_coords, method="linear", fill_value=fill
    )

    return np.stack([inv_i.reshape(H, W), inv_j.reshape(H, W)], axis=0)
