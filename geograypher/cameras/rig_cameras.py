from pathlib import Path
from typing import Dict, List

from geograypher.cameras import MetashapeCameraSet, PhotogrammetryCameraSet
from geograypher.constants import PATH_TYPE
from geograypher.utils.image import rotate_by_roll_pitch_yaw


def create_rig_cameras_from_equirectangular(
    camera_file: PATH_TYPE,
    original_images: PATH_TYPE,
    perspective_images: PATH_TYPE,
    rig_camera: Dict[str, float],
    rig_orientations: List[Dict[str, float]],
    perspective_filename_format_str: str,
) -> PhotogrammetryCameraSet:
    """
    Create a synthetic camera set corresponding to a rig of perspective cameras derived from an
    equirectangular camera. This is a workaround for directly projecting an equirectangular
    representation because 1) the rendering libraries do not support equirectangular models and 2)
    it's often easier to do prediction tasks on a perspective crop.

    Args:
        camera_file (PATH_TYPE):
            Path to a metashape results file from the real equirectangular images.
        original_images (PATH_TYPE):
            The location of equirectangular images used to generate the photogrammetry products
        perspective_images (PATH_TYPE):
            The resampled perspective images obtained from the equirectangular images. It's expected
            that each image filename contains information about the original image it was generated
            from and then the orientation it was captured in.
        rig_camera (Dict[str, float]):
            A dict with "f", "cx", "cy", "image_width" and "image_height" keys. This defines the
            parameters of the synthetic camera. Note that since this is synthetic, it's common for
            cx and cy to be zero.
        rig_orientations (List[Dict[str, float]]):
            A list of dicts where each one has the keys "roll_deg", "pitch_deg", and "yaw_deg".
            Each dict indicates the rotation of a given camera in the rig. The center of the
            equirectangular image corresponds to zero roll.
        perspective_filename_format_str (str):
            A format string that accepts named arguments "roll_deg", "pitch_deg", and "yaw_deg". The
            result of this string should be concatenated to the original equirectangular image
            filename (before the extension) to obtain the filename of the perspective image.

    Returns:
        PhotogrammetryCameraSet: A camera set representing the synthetic rig.
    """
    # Load the camera file. This just serves as an easy way to parse the required information,
    # it will not be used directly. The sensor params are defaulted since the spherical camera does
    # not have these attributes but they are required for loading the data.
    initial_camera_set = MetashapeCameraSet(
        camera_file=camera_file,
        image_folder=perspective_images,
        original_image_folder=original_images,
        default_sensor_params={"f": 1.0, "cx": 0.0, "cy": 0.0},
    )

    # Extract the attributes from the original camera set
    cam_to_world_transforms = [
        c.cam_to_world_transform for c in initial_camera_set.cameras
    ]
    image_filenames = [c.image_filename for c in initial_camera_set.cameras]

    # Compute the rotation transforms for each of the cameras in the rig
    rig_transforms = [
        rotate_by_roll_pitch_yaw(**rig_orientation, return_4x4=True)
        for rig_orientation in rig_orientations
    ]

    # Compute the extension string for each image
    image_extensions = [
        perspective_filename_format_str.format(**rig_orientation)
        for rig_orientation in rig_orientations
    ]

    # Apply each of the rig transforms to each transform in the cameras set
    new_transforms = [
        cam_to_world @ rig_transform
        for cam_to_world in cam_to_world_transforms
        for rig_transform in rig_transforms
    ]
    # Determine the new image paths
    new_image_filenames = [
        Path(
            image_filename.parent,
            image_filename.stem + image_extension + ".png",
        )
        for image_filename in image_filenames
        for image_extension in image_extensions
    ]
    # TODO consider supporting multiple sensors within a rig. Though it's a synthetic rig so that
    # might not ever be needed.
    sensor_ids = [0] * len(new_image_filenames)

    # Use this information to create a new camera set that has one element for each camera in the
    # synthetic rig.
    rig_cameras = PhotogrammetryCameraSet(
        cam_to_world_transforms=new_transforms,
        intrinsic_params_per_sensor_type={0: rig_camera},
        image_filenames=new_image_filenames,
        sensor_IDs=sensor_ids,
        local_to_epsg_4978_transform=initial_camera_set.get_local_to_epsg_4978_transform(),
    )

    return rig_cameras
