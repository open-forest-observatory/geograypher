from geograypher.constants import PATH_TYPE
from typing import List, Dict

from geograypher.cameras import MetashapeCameraSet, PhotogrammetryCameraSet
from geograypher.utils.image import rotate_by_roll_pitch_yaw
from pathlib import Path


def create_rig_cameras(
    camera_file: PATH_TYPE,
    original_images: PATH_TYPE,
    resampled_images: PATH_TYPE,
    rig_camera: Dict[str, float],
    rig_orientations: List[Dict[str, float]],
    resampled_filename_format_str: str,
):
    initial_camera_set = MetashapeCameraSet(
        camera_file=camera_file,
        image_folder=resampled_images,
        original_image_folder=original_images,
        default_sensor_params={"f": 1.0, "cx": 0.0, "cy": 0.0},
    )

    # Extract the attributes from the original camera set
    cam_to_world_transforms = [
        c.cam_to_world_transform for c in initial_camera_set.cameras
    ]
    image_filenames = [c.image_filename for c in initial_camera_set.cameras]

    # Compute the transforms
    rig_transforms = [
        rotate_by_roll_pitch_yaw(**rig_orientation, return_4x4=True)
        for rig_orientation in rig_orientations
    ]

    image_extensions = [
        resampled_filename_format_str.format(**rig_orientation)
        for rig_orientation in rig_orientations
    ]

    # Lumped by cameras (this changes slower)
    new_transforms = [
        cam_to_world @ rig_transform
        for cam_to_world in cam_to_world_transforms
        for rig_transform in rig_transforms
    ]
    new_image_filenames = [
        Path(
            image_filename.parent,
            image_filename.stem + image_extension + ".png",
        )
        for image_filename in image_filenames
        for image_extension in image_extensions
    ]
    sensor_ids = [0] * len(new_image_filenames)

    cam_set = PhotogrammetryCameraSet(
        cam_to_world_transforms=new_transforms,
        intrinsic_params_per_sensor_type={0: rig_camera},
        image_filenames=new_image_filenames,
        sensor_IDs=sensor_ids,
        local_to_epsg_4978_transform=initial_camera_set.get_local_to_epsg_4978_transform(),
    )

    return cam_set
