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


def rotate_by_roll_pitch_yaw(
    roll_deg: float,
    pitch_deg: float,
    yaw_deg: float,
    return_4x4: bool = False,
) -> np.ndarray:
    # Convert from degrees to radians
    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)
    roll = np.deg2rad(roll_deg)

    # RPY is defined with roll about X, pitch about Y, yaw about Z. We need to convert from the
    # perspective camera frame axes convention to the RPY convention.
    # New Z has to be old -Y
    # New Y has to be old X
    # New X has to be old Z
    permutation_matrix = np.array([[0, 0, 1], [1, 0, 0], [0, -1, 0]])

    # Calculate the rotation matrix corresponding to the roll-pitch-yaw convention
    # https://stackoverflow.com/questions/74434119/scipy-rotation-matrix-from-as-euler-angles
    rotation_matrix = Rotation.from_euler("ZYX", [yaw, pitch, roll]).as_matrix()

    # The permutation matrix can be thought of first converting into the conventional RPY frame,
    # then applying the rotation, then converting back into the camera frame.
    rotation_matrix_in_cam_frame = (
        permutation_matrix.T @ rotation_matrix @ permutation_matrix
    )

    if return_4x4:
        # Pad with zeros except for a 1 at 3,3
        rotation_matrix_in_cam_frame = np.concatenate(
            [
                np.concatenate(
                    [rotation_matrix_in_cam_frame, np.zeros((3, 1))], axis=1
                ),
                np.array([[0, 0, 0, 1]]),
            ],
            axis=0,
        )

    return rotation_matrix_in_cam_frame


def flexible_inputs_warp(
    input_image: np.ndarray,
    inverse_map: np.ndarray,
    interpolation_order: int = None,
    fill_value: float = 0.0,
) -> np.ndarray:
    """
    Extends the functionality of skimage.transform.warp to handle arbitrary datatypes and
    multi-channel images
    """
    # Temporarily expand grayscale images to be (N, M, 1)
    input_image = np.atleast_3d(input_image)

    # Note if the fill_value is not actually used and is in fact outside the range of the input
    # values it will result in an artificially small range of data for the real values. But it won't
    # affect the results aside from the possible loss of precision.
    # Compute the min and max of values, including both the inputs and the fill value
    input_min = min(np.min(input_image), fill_value)
    input_max = max(np.max(input_image), fill_value)
    # Compute the range of values
    min_max_range = input_max - input_min

    # If there's no variation, we can avoid division by 0 issues and save computation by just
    # returning an array of one value
    if min_max_range == 0:
        return np.full_like(np.squeeze(input_image), fill_value=fill_value)

    # Record the original datatype
    initial_dtype = input_image.dtype
    # Rescale to 0-1
    input_image = (input_image.astype(float) - input_min) / min_max_range
    # The fill value will be applied to the rescaled image, so it must be rescaled similarly
    rescaled_fill_value = (float(fill_value) - input_min) / min_max_range

    # Create an output image that's the shape of inverse map (minus first dimension which is 2 [i,j])
    # and the number of channels from the input image
    output_image = np.zeros(inverse_map.shape[1:] + (input_image.shape[2],))
    # For each color channel, do the warping
    for channel in range(input_image.shape[2]):
        warped = warp(
            image=input_image[:, :, channel],
            inverse_map=inverse_map,
            order=interpolation_order,  # interpolation strategy for fractional pixels
            mode="constant",  # fill unseen areas with cval
            cval=rescaled_fill_value,
            clip=True,  # clip to [0,1]
            preserve_range=True,  # keep original range without rescaling
        )
        output_image[:, :, channel] = warped

    # Convert to the original range and datatype
    output_image = ((output_image * min_max_range) + input_min).astype(initial_dtype)

    # If the original image was grayscale, remove the added singleton dimension
    return np.squeeze(output_image)


def perspective_from_equirectangular(
    equi_img: np.ndarray,
    fov_deg: float,
    output_size: Tuple[int, int] = (1440, 1440),
    yaw_deg: float = 0,
    pitch_deg: float = 0,
    roll_deg: float = 0,
    warp_order: int = 1,
    oversample_factor: int = 1,
    return_mask: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Sample a perspective image from an equirectangular (360) image. Different parameters of the
    "virtual camera" can be controlled, including the field of view, orientation (yaw, pitch, roll),
    and output image size.

    The rotations are defined using the roll-pitch-yaw convention. The roll axis corresponds to the
    camera Z (forward) axis, pitch corresponds to the X axis, and yaw corresponds to the Y axis.

    With RPY all set to zero, the camera is looking toward the center of the equirectangular image.

    Args:
        equi_img (np.ndarray):
            Equirectangular image in (H, W, C) format.
        fov_deg float: camera horizontal field of view.
        output_size Tuple[int, int]: Shape of the output image to sample (i, j).
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
    out_h, out_w = output_size

    # Sample a larger image and then downsample to reduce aliasing
    out_w = int(out_w * oversample_factor)
    out_h = int(out_h * oversample_factor)

    # FOV and angles in radians
    fov = np.deg2rad(fov_deg)

    # Compute the aspect ratio of the requested image dimensions
    aspect_ratio = out_h / out_w

    # Compute the distance from the center of the image to the edge in the x dimension
    x_dist = np.tan(fov / 2)
    y_dist = x_dist * aspect_ratio

    # Compute the size of each pixel to center the rays within the pixel rather than using the
    # top-left corner
    pixel_width = (2 * x_dist) / out_w

    # Create homogenous image coordinates for the output image
    x = np.arange(-x_dist + pixel_width / 2, x_dist, pixel_width)
    y = np.arange(-y_dist + pixel_width / 2, y_dist, pixel_width)

    # invert y for image coordinates to account for array indexing notation
    xv, yv = np.meshgrid(x, -y)

    # The z direction
    zv = np.ones_like(xv)

    pixel_directions = np.stack((xv, yv, zv), axis=-1)
    # normalize to unit
    pixel_directions /= np.linalg.norm(pixel_directions, axis=-1, keepdims=True)

    rotation_matrix = rotate_by_roll_pitch_yaw(roll_deg, pitch_deg, yaw_deg)

    # Rotate the pixel directions by the rotation matrix
    # The strange convention here is to deal with the fact that pixel_directions is (w, h, 3)
    # so this allows the dimensions to align for the matrix multiplication.
    pixel_directions = pixel_directions @ rotation_matrix.T

    # Convert 3D directions to spherical coordinates
    # horizontal angle
    horizontal = np.arctan2(pixel_directions[..., 0], pixel_directions[..., 2])
    # vertical angle. Clip to avoid floating point errors which extend beyond the valid domain
    # of arcsin
    altitude = np.arcsin(np.clip(pixel_directions[..., 1], -1.0, 1.0))

    # Map to equirectangular image coordinates
    i = (0.5 - altitude / np.pi) * H
    j = (horizontal / (2 * np.pi) + 0.5) * W
    # The range of these coordinates would be i in [0, H], j in [0, W]
    # To account for the wraparound effects if the sampling goes of the right edge, add the left row
    # of pixels to the right side.
    # TODO think more about sub-pixel considerations and how these coordinates interact with skimage
    # conventions on sampling and interpolation.
    equi_img = np.concatenate([equi_img, equi_img[:, 0:1, :]], axis=1)

    # Ensure that the coordinates are within the image. Note that the cropping in j is more generous
    # because of the wrap-around pixel.
    i = np.clip(i, 0, H - 1)
    j = np.clip(j, 0, W)

    # Stack the coordinates
    ij = np.stack((i, j), axis=0)

    # Sample pixels from the specified coordinates to obtain the perspective image
    sampled_perspective = flexible_inputs_warp(
        input_image=equi_img, inverse_map=ij, interpolation_order=warp_order
    )

    if oversample_factor > 1:
        sampled_perspective = downscale_local_mean(
            sampled_perspective, (oversample_factor, oversample_factor, 1)
        )

    # If only the resampled image is needed, return it
    if not return_mask:
        return sampled_perspective
    # Otherwise return a mask of the pixels which were sampled

    # Also save a mask of the pixels being sampled.
    mask = np.zeros((equi_img.shape[0], equi_img.shape[1]), dtype=bool)
    # Set pixels which were sampled to True. Out of bound errors are not possible because the values
    # have been clipped to the appropriate range.
    mask[np.round(i).astype(int), np.round(j).astype(int)] = True

    # In some cases the right edge might have been sampled, so report this as a sampled pixel on the
    # left edge
    mask[:, 0] = np.logical_or(mask[:, 0], mask[:, -1])
    # Remove the right edge pixels
    mask = mask[:, :-1]

    return sampled_perspective, mask
