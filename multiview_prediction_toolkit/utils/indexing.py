import numpy as np
import pandas as pd
import logging


def ensure_float_labels(query_array, full_array=None) -> np.ndarray:
    # Ensure that the query is a numpy array
    if isinstance(query_array, pd.Series):
        query_array = query_array.to_numpy()
    else:
        query_array = np.array(query_array)

    # Check if the array is a string or some other object type that is not numeric
    if query_array.dtype in (str, np.dtype("O")):
        # Get unique values from full array
        if full_array is None:
            # This means that indices could change based on the subset that is selected
            full_array = query_array

        # These values will always be sorted
        # This works even if full_array is a pd.Series, so we don't type check it
        unique_values = np.unique(full_array)
        logging.warn(
            f"Creating categorical labels with the following classes {unique_values}"
        )

        # Build an output array
        output_query_array = np.full_like(query_array, fill_value=np.nan, dtype=float)

        # Iterate through and set the values that match to the integer label
        for i, unique_value in enumerate(unique_values):
            output_query_array[query_array == unique_value] = i
        # Set the output array
    else:
        # Not meaningful here
        unique_values = None
        output_query_array = query_array.astype(float)

    return output_query_array, unique_values
