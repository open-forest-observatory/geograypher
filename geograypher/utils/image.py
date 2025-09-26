import piexif
from PIL import Image

import numpy as np
from PIL import Image
import piexif
from skimage.transform import warp, downscale_local_mean
from scipy.spatial.transform import Rotation


def get_GPS_exif(filename):
    im = Image.open(filename)
    exif_dict = piexif.load(im.info["exif"])
    lat = exif_dict["GPS"][piexif.GPSIFD.GPSLatitude]
    lon = exif_dict["GPS"][piexif.GPSIFD.GPSLongitude]
    lat = (
        lat[0][0] / lat[0][1]
        + lat[1][0] / (lat[1][1] * 60)
        + lat[2][0] / (lat[2][1] * 3600)
    )
    lon = (
        lon[0][0] / lon[0][1]
        + lon[1][0] / (lon[1][1] * 60)
        + lon[2][0] / (lon[2][1] * 3600)
    )
    # https://stackoverflow.com/questions/54196347/overwrite-gps-coordinates-in-image-exif-using-python-3-6
    return lon, lat


def perspective_from_equirectangular(
    equi_img,
    fov_deg=90,
    yaw_deg=0,
    pitch_deg=0,
    output_size=(1440, 1440),
    warp_order: int = 1,
    oversample_factor: int = 1,
    return_mask: bool = False,
):
    """Convert equirectangular (360) image to a perspective view.
    Parameters:
    - equi_img: np.ndarray the equirectangular image
    - fov_deg: float, horizontal field of view in degrees
    - yaw_deg: float, rotation around vertical axis (degrees)
    - pitch_deg: float, rotation around horizontal axis (degrees)
    - warp_order: int, order of polynomal interpolation in skimage.transform.warp
    - oversample_factor: int, sample an image from the original which is larger by this factor to avoid aliasing

    - output_size: (width, height) of output perspective image. Defaults to 1/4 the width.
    Returns:
    - PIL.Image of perspective projection
    """
    H, W = equi_img.shape[:2]

    # Output image dimensions
    out_w, out_h = output_size

    # Sample a larger image and then downsample to reduce aliasing
    out_w = int(out_w * oversample_factor)
    out_h = int(out_h * oversample_factor)

    # FOV and angles in radians
    fov = np.deg2rad(fov_deg)
    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)

    # Compute the aspect ratio of the requested image dimensions
    aspect_ratio = out_h / out_w

    # Create homogenous image coordinates for the output image
    x = np.linspace(-np.tan(fov / 2), np.tan(fov / 2), out_w)
    y = np.linspace(
        -np.tan(fov / 2) * aspect_ratio, np.tan(fov / 2) * aspect_ratio, out_h
    )

    xv, yv = np.meshgrid(
        x, -y
    )  # invert y for image coordinates to account for array indexing notation

    # The z direction
    zv = np.ones_like(xv)

    pixel_directions = np.stack((xv, yv, zv), axis=-1)
    # normalize to unit
    pixel_directions /= np.linalg.norm(pixel_directions, axis=-1, keepdims=True)

    rotation_matrix = Rotation.from_euler("yx", [yaw, pitch]).as_matrix()

    # Rotate the pixel directions by the rotation matrix
    # The strange convention here is to deal with the fact that pixel_directions is (w, h, 3)
    # so this allows the dimensions to align for the matrix multiplication.
    pixel_directions = pixel_directions @ rotation_matrix.T

    # Convert 3D directions to spherical coordinates
    # horizontal angle
    horizontal = np.arctan2(pixel_directions[..., 0], pixel_directions[..., 2])
    # vertical angle
    altitude = np.arcsin(pixel_directions[..., 1])

    # Map to equirectangular image coordinates
    i = (0.5 - altitude / np.pi) * H
    j = (horizontal / (2 * np.pi) + 0.5) * W

    # Replicate the pixels from the opposite side of the image to handle wrap-around
    equi_img = np.concatenate(
        [equi_img[:, -1:, :], equi_img, equi_img[:, 0:1, :]], axis=1
    )

    j += 1  # offset by 1 to account for the extra pixel at the start

    # Ensure that the coordinates are within the image. Note that the cropping in j is more generous
    # becuase of the wrap-around pixels.
    i = np.clip(i, 0, H - 1)
    j = np.clip(j, 0, W + 1)

    # Stack the coordinates
    ij = np.stack((i, j), axis=0)

    # Sample pixels from the specified coordinates to obtain the perspective image
    # Note this must be done channel-wise
    sampled_perspective = np.stack(
        [
            warp(equi_img[..., c], ij, order=warp_order)
            for c in range(equi_img.shape[2])
        ],
        axis=2,
    )

    if oversample_factor > 1:
        sampled_perspective = downscale_local_mean(
            sampled_perspective, (oversample_factor, oversample_factor, 1)
        )

    # If only the resampled image is needed, return it
    if not return_mask:
        return sampled_perspective
    # Otherwise return a mask of the pixels which were sampled

    # Also save a mask of the pixels being sampled
    mask = np.zeros(equi_img.shape[:2], dtype=bool)
    mask[i.astype(int), j.astype(int)] = True

    # Handle edge effects by logical ORing the original edge with the opposite side
    mask[:, 1] = np.logical_or(mask[:, 1], mask[:, -1])
    mask[:, -2] = np.logical_or(mask[:, -2], mask[:, 0])
    # Crop the additional edge pixels
    mask = mask[:, 1:-1]

    return sampled_perspective, mask
