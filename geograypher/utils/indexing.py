import logging

import numpy as np
import pandas as pd


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
