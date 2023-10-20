import argparse
import logging
from tqdm import tqdm
import numpy as np
from pathlib import Path

from multiview_prediction_toolkit.cameras import MetashapeCameraSet
from multiview_prediction_toolkit.config import (
    DATA_FOLDER,
    DEFAULT_CAM_FILE,
    DEFAULT_GEOPOLYGON_FILE,
    DEFAULT_IMAGES_FOLDER,
    DEFAULT_LOCAL_MESH,
    PATH_TYPE,
)
from multiview_prediction_toolkit.meshes import TexturedPhotogrammetryMesh


def parse_args():
    """Parse and return arguements

    Returns:
        argparse.Namespace: Arguments
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--mesh-file", default=DEFAULT_LOCAL_MESH)
    parser.add_argument("--camera-file", default=DEFAULT_CAM_FILE)
    parser.add_argument("--image-folder", default=DEFAULT_IMAGES_FOLDER)
    parser.add_argument("--render-folder", default="data/gascola/renders_new")
    parser.add_argument("--mesh-downsample", type=float, default=1)
    parser.add_argument("--image-downsample", type=float, default=0.25)
    parser.add_argument("--vector-file", default=DEFAULT_GEOPOLYGON_FILE)
    parser.add_argument("--vector-file-column", default="treeID")
    parser.add_argument("--vis", action="store_true")
    parser.add_argument(
        "--screenshot-filename",
    )
    parser.add_argument(
        "--log-level",
        default="info",
        choices=list(logging._nameToLevel.keys()),
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    logging.basicConfig(level=args.log_level.upper())

    # Load the camera set
    logging.info("Creating the camera set")
    camera_set = MetashapeCameraSet(args.camera_file, args.image_folder)

    # Load the mesh
    logging.info("Loading the mesh")
    mesh = TexturedPhotogrammetryMesh(
        args.mesh_file, downsample_target=args.mesh_downsample
    )

    mesh.get_values_for_verts_from_vector(
        column_names=args.vector_file_column,
        vector_file=args.vector_file,
        set_vertex_IDs=True,
    )

    if args.vis or args.screenshot_filename is not None:
        mesh.vis(screenshot_filename=args.screenshot_filename)

    for i in tqdm(range(camera_set.n_cameras())):
        image = camera_set.get_image_by_index(i, image_scale=args.image_downsample)
        image_path = camera_set.get_camera_by_index(i).image_filename
        label_mask = mesh.render_pytorch3d(
            camera_set, image_scale=args.image_downsample, camera_index=i
        )
        savepath = Path(
            DATA_FOLDER,
            args.render_folder,
            str(Path(image_path)).replace(".jpg", ".npy"),
        )

        savepath.parent.mkdir(parents=True, exist_ok=True)
        np.save(
            savepath,
            label_mask,
        )
