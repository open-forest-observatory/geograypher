import numpy as np
import pytest

from geograypher.utils.indexing import inverse_map_interpolation


@pytest.mark.parametrize("downsample", [1, 2])
def test_inverse_map_interpolation(downsample):
    """Test that a simple mapping can be reversed."""

    # For i, we are sampling to i+2. When we reverse this, we should be
    # sampling to i-2. Note that invalid values should all be -1 due to the
    # fill argument.
    imap = np.array(
        [
            [2, 2, 2, 2, 2, 2],
            [3, 3, 3, 3, 3, 3],
            [4, 4, 4, 4, 4, 4],
            [5, 5, 5, 5, 5, 5],
            [6, 6, 6, 6, 6, 6],
        ]
    )
    # For j, we are mostly sampling to the same j. However, the last column
    # takes a j+1 jump, so when inverting the last column will be an average
    # of the last two.
    jmap = np.array(
        [
            [0, 1, 2, 3, 4, 6],
            [0, 1, 2, 3, 4, 6],
            [0, 1, 2, 3, 4, 6],
            [0, 1, 2, 3, 4, 6],
            [0, 1, 2, 3, 4, 6],
        ]
    )
    ijmap = np.stack([imap, jmap], axis=0)
    inv_imap, inv_jmap = inverse_map_interpolation(ijmap, downsample=downsample)

    # Account for how downsample 2 affects things
    i_expected = np.array(
        [
            [-1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1],
            [0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1],
            [2, 2, 2, 2, 2, 2],
        ]
    )
    j_expected = np.array(
        [
            [-1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1],
            [0, 1, 2, 3, 4, 4.5],
            [0, 1, 2, 3, 4, 4.5],
            [0, 1, 2, 3, 4, 4.5],
        ]
    )
    if downsample == 2:
        i_expected[:, -1] = -1
        j_expected[:, -1] = -1
    assert np.allclose(inv_imap, i_expected)
    assert np.allclose(inv_jmap, j_expected)
