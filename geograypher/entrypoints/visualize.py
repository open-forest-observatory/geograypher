import argparse

import numpy as np

from geograypher.cameras import MetashapeCameraSet
from geograypher.meshes import TexturedPhotogrammetryMesh


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh-file")
    parser.add_argument("--camera-file")
    parser.add_argument("--texture")
    parser.add_argument("--downsample-target", type=float, default=1.0)
    parser.add_argument("--ROI")
    parser.add_argument("--ROI-buffer-meters", type=float, default=0.0)
    parser.add_argument("--convert-texture-to-max-class", action="store_true")

    args = parser.parse_args()
    return args


def visualize(
    mesh_file,
    camera_file,
    texture,
    downsample_target,
    ROI,
    ROI_buffer_meters,
    convert_texture_to_max_class,
):
    if convert_texture_to_max_class:
        texture = np.load(texture)
        max_class = np.argmax(texture, axis=1).astype(float)
        proj_per_face = np.sum(texture, axis=1)
        invalid_faces = np.logical_or(
            proj_per_face == 0, np.logical_not(np.isfinite(proj_per_face))
        )

        max_class[invalid_faces] = np.nan
        texture = max_class

    # Load the camera set if provided
    if camera_file is not None:
        camera_set = MetashapeCameraSet(camera_file=camera_file, image_folder="")
    else:
        camera_set = None

    mesh = TexturedPhotogrammetryMesh(
        mesh_file,
        ROI=ROI,
        ROI_buffer_meters=ROI_buffer_meters,
        downsample_target=downsample_target,
        texture=texture,
    )

    mesh.vis(camera_set=camera_set)


if __name__ == "__main__":
    args = parse_args()
    visualize(**args.__dict__)
