import numpy as np
from geograypher.utils.image import perspective_from_equirectangular
import matplotlib.pyplot as plt
import pytest


@pytest.mark.parametrize(
    "expected,yaw_deg,pitch_deg",
    (
        [[(45, 270), (90, 315), (135, 270), (90, 225)], 90, 0],
        [[(45, 360), (90, 45), (135, 360), (90, 315)], 180, 0],
    ),
)
def test_equi_to_perspective(expected, yaw_deg, pitch_deg):
    output_size = (91, 91)
    fov_deg = 90
    oversample_factor = 1
    warp_order = 0

    # Create a test equirectangular image where the pixel values are the (i,j) coordinates
    i_samples = np.arange(180)
    j_samples = np.arange(360)

    ij = np.meshgrid(i_samples, j_samples, indexing="ij")
    img = np.stack(
        ij,
        axis=-1,
    )
    # Divide by 360 to get values in [0, 1] for skimage
    img = img / 360.0

    sample, _ = perspective_from_equirectangular(
        equi_img=img,
        fov_deg=fov_deg,
        yaw_deg=yaw_deg,
        pitch_deg=pitch_deg,
        output_size=output_size,
        warp_order=warp_order,
        oversample_factor=oversample_factor,
    )
    # Re-scale back to original pixel values
    sample = sample * 360.0

    # Check the four pixels at the center of each side to see if the value is what was expected
    assert np.allclose(sample[0, 45, :2], expected[0], atol=1)
    assert np.allclose(sample[45, 90, :2], expected[1], atol=1)
    assert np.allclose(sample[90, 45, :2], expected[2], atol=1)
    assert np.allclose(sample[45, 0, :2], expected[3], atol=1)
