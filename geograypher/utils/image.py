from typing import Optional, Tuple

import numpy as np
import piexif
from PIL import Image
from scipy.spatial.transform import Rotation
from skimage.transform import downscale_local_mean, warp


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
    equi_img: np.ndarray,
    fov_deg: float,
    output_size: Tuple[float] = (1440, 1440),
    yaw_deg: float = 0,
    pitch_deg: float = 0,
    roll_deg: float = 0,
    warp_order: int = 1,
    oversample_factor: int = 1,
    return_mask: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Sample a perspective image from an equirectangular (360) image. Different parameters of the
    "virtual camera" can be controlled, including the field of view, oreintation (yaw, pitch, roll),
    and output image size.

    Args:
        equi_img (np.ndarray):
            Equirectangular image in (H, W, C) format. Currently only float (0,1) images are
            supported but this is a TODO.
        fov_deg float: camera horizontal field of view.
        output_size Tuple[float]: Shape of the output image to sample (i, j).
        yaw_deg (float, optional): yaw camera orientation. Defaults to 0.
        pitch_deg (float, optional): pitch camera orientation. Defaults to 0.
        roll_deg (float, optional): roll camera orientation. Defaults to 0.
        warp_order (int, optional): Interpolation order to use when sampling pixels. Defaults to 1.
        oversample_factor (int, optional):
            Sample this number times more pixels in each dimension. This helps avoid aliasing but
            increases runtime. Defaults to 1.
        return_mask (bool, optional):
           Return a mask showing what pixels from the original image were
             sampled. Defaults to False.

    Returns:
        np.ndarray: The sampled perspective image
        Optional[np.ndarray]]: The mask of sampled locations, if requested
    """
    # Convert an equirectangular (360) image to a perspective view.
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
    roll = np.deg2rad(roll_deg)

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

    # Seems to correspond to the roll-pitch-yaw convention
    rotation_matrix = Rotation.from_euler("zyx", [roll, yaw, pitch]).as_matrix()

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
