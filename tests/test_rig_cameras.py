import pytest
from geograypher.cameras.rig_cameras import create_rig_cameras
from pathlib import Path
import numpy as np


def test_rig_cameras():
    CAMERAS_FILENAME = Path(
        "/ofo-share/scratch-eric/tmp/automate-test-360-st0077_SHIFT/automate-test-360-st0077_cameras.xml"
    )
    IMAGE_FOLDER = Path(
        "/ofo-share/scratch-eric/exp360/2024-ofo-gopro__st-0077-gopro-360-photos-timelapse/"
    )
    PREDICTED_IMAGE_LABELS_FOLDER = Path(
        "/ofo-share/repos-david/under-canopy-mapping/data/predictions/2024-ofo-gopro__st-0077-gopro-360-photos-timelapse-subset"
    )
    RIG_CAMERA_DEF = {
        "f": 1440 / np.sqrt(2),
        "cx": 0.0,
        "cy": 0.0,
        "image_width": 1440,
        "image_height": 1440,
    }
    RIG_ORIENTATIONS = [
        {"yaw_deg": 0.0, "pitch_deg": 0.0, "roll_deg": 0.0},
        {"yaw_deg": 90.0, "pitch_deg": 0.0, "roll_deg": 0.0},
        {"yaw_deg": 180.0, "pitch_deg": 0.0, "roll_deg": 0.0},
        {"yaw_deg": 270.0, "pitch_deg": 0.0, "roll_deg": 0.0},
        {"yaw_deg": 0.0, "pitch_deg": -90.0, "roll_deg": 0.0},
        {"yaw_deg": 0.0, "pitch_deg": 90.0, "roll_deg": 0.0},
    ]
    RESAMPLED_FORMAT_STR = "_yaw{yaw_deg:.0f}_pitch{pitch_deg:.0f}"

    create_rig_cameras(
        camera_file=CAMERAS_FILENAME,
        original_images=IMAGE_FOLDER,
        resampled_images=PREDICTED_IMAGE_LABELS_FOLDER,
        rig_camera=RIG_CAMERA_DEF,
        rig_orientations=RIG_ORIENTATIONS,
        resampled_filename_format_str=RESAMPLED_FORMAT_STR,
    )
