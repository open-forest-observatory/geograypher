import matplotlib.pyplot as plt
import numpy as np
import pytest

from geograypher.utils.image import perspective_from_equirectangular


def convert_py_to_xyz(pitch_yaw_deg):
    """Take a pitch-yaw representation and convert it to a direction on a unit sphere

    Args:
        pitch_yaw_deg (tuple[float]): pitch and yaw in degrees

    Returns:
        np.array: [x,y,z] unit vector
    """
    pitch_rad = np.deg2rad(pitch_yaw_deg[0])
    yaw_rad = np.deg2rad(pitch_yaw_deg[1])

    # x is the direction of the initial forward direction
    x = np.cos(pitch_rad) * np.cos(yaw_rad)
    # y is the left direction with the initial direction being forward
    y = np.cos(pitch_rad) * np.sin(yaw_rad)
    # z is the up direction with the initial direction being forward
    z = np.sin(pitch_rad)

    return np.array([x, y, z])


# @pytest.mark.parametrize(
#    "yaw_deg,pitch_deg,roll_deg",
#    ([180, 90, 30],),
# )
@pytest.mark.parametrize("yaw_deg", [0, 45, 180, 270])
@pytest.mark.parametrize("pitch_deg", [0, 45, 90, -90])
@pytest.mark.parametrize("roll_deg", [0, 30, 45, 180])
def test_equi_to_perspective(yaw_deg, pitch_deg, roll_deg):
    output_size = (91, 91)
    fov_deg = 60
    oversample_factor = 1
    warp_order = 0

    # Create a test equirectangular image where the pixel values are the (pitch, yaw) coordinates,
    # Defined using the center of the image as (0,0)

    # Negative from image convention
    i_samples = np.arange(90, -90, -1)
    j_samples = np.arange(-180, 180, 1)

    ij = np.meshgrid(i_samples, j_samples, indexing="ij")
    img = np.stack(
        ij,
        axis=-1,
    )
    # scale and shift to (0,1) range for skimage compatibility
    img = (img / 360.0) + 0.5

    sample, mask = perspective_from_equirectangular(
        equi_img=img,
        fov_deg=fov_deg,
        yaw_deg=yaw_deg,
        pitch_deg=pitch_deg,
        roll_deg=roll_deg,
        output_size=output_size,
        warp_order=warp_order,
        oversample_factor=oversample_factor,
        return_mask=True,
    )
    # Re-scale back to original pixel values
    sample = sample * 360.0 - 180.0

    # Check the four pixels at the center of each side to see if the value is what was expected
    # assert np.allclose(sample[0, 45, :2], expected[0], atol=1)
    # assert np.allclose(sample[45, 90, :2], expected[1], atol=1)
    # assert np.allclose(sample[90, 45, :2], expected[2], atol=1)
    # assert np.allclose(sample[45, 0, :2], expected[3], atol=1)

    xyz_sample = convert_py_to_xyz(sample[45, 45, :])
    xyz_input = convert_py_to_xyz([pitch_deg, yaw_deg])
    print(
        np.array(xyz_sample) - np.array(xyz_input),
        sample[45, 45, :],
        [pitch_deg, yaw_deg],
    )
    assert np.allclose(
        xyz_sample,
        xyz_input,
        atol=0.02,  # TODO decrease this tolerance
    )

    ## For debugging, uncomment these
    ## This shows the indices in the original image from which the perspective image was sampled
    if False:
        f, ax = plt.subplots(1, 3)
        cb = ax[0].imshow(sample[:, :, 0])
        f.colorbar(cb, ax=ax[0])
        ax[0].title.set_text("I values")

        cb = ax[1].imshow(sample[:, :, 1])
        f.colorbar(cb, ax=ax[1])
        ax[1].title.set_text("J values")

        cb = ax[2].imshow(mask)
        f.colorbar(cb, ax=ax[2])
        ax[2].title.set_text("Mask")
        plt.show()
