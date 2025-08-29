import numpy as np

from geograypher.utils.indexing import inverse_map_interpolation


def test_inverse_map_interpolation():
    """Test that a simple mapping can be reversed."""

    # For i, we are sampling to i+2. When we reverse this, we should be
    # sampling to i-2. Note that invalid values should all be -1 due to the
    # fill argument.
    imap = np.array(
        [
            [2, 2, 2, 2, 2],
            [3, 3, 3, 3, 3],
            [4, 4, 4, 4, 4],
            [5, 5, 5, 5, 5],
            [6, 6, 6, 6, 6],
        ]
    )
    # For j, we are mostly sampling to the same j. However, the last column
    # takes a j+1 jump, so when inverting the last column will be an average
    # of the last two.
    jmap = np.array(
        [
            [0, 1, 2, 3, 5],
            [0, 1, 2, 3, 5],
            [0, 1, 2, 3, 5],
            [0, 1, 2, 3, 5],
            [0, 1, 2, 3, 5],
        ]
    )
    ijmap = np.stack([imap, jmap], axis=0)
    inv_imap, inv_jmap = inverse_map_interpolation(ijmap)
    assert np.allclose(
        inv_imap,
        np.array(
            [
                [-1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1],
                [0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1],
                [2, 2, 2, 2, 2],
            ]
        ),
    )
    assert np.allclose(
        inv_jmap,
        np.array(
            [
                [-1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1],
                [0, 1, 2, 3, 3.5],
                [0, 1, 2, 3, 3.5],
                [0, 1, 2, 3, 3.5],
            ]
        ),
    )
