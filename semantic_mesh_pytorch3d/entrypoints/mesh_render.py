import argparse
from pathlib import Path

from semantic_mesh_pytorch3d.config import (
    DEFAULT_CAM_FILE,
    DEFAULT_IMAGES_FOLDER,
    DEFAULT_LOCAL_MESH,
)
from semantic_mesh_pytorch3d.meshes import Pytorch3DMesh


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--mesh-file", default=DEFAULT_LOCAL_MESH)
    parser.add_argument("--camera-file", default=DEFAULT_CAM_FILE)
    parser.add_argument("--image-folder", default=DEFAULT_IMAGES_FOLDER)
    parser.add_argument(
        "--texture-type",
        default=0,
        type=int,
        help="Enum for texture. 0: default texture, 1: dummy texture, 2: geofile texture",
    )
    parser.add_argument(
        "--run-vis", action="store_true", help="Run mesh and cameras visualization"
    )
    parser.add_argument(
        "--run-aggregation",
        action="store_true",
        help="Aggregate color from multiple viewpoints",
    )
    parser.add_argument(
        "--run-render", action="store_true", help="Render out viewpoints"
    )
    args = parser.parse_args()
    return args


def main(
    mesh_file,
    camera_file,
    image_folder,
    run_vis,
    run_aggregation,
    run_render,
    texture_type,
):
    mesh = Pytorch3DMesh(
        mesh_file, camera_file, image_folder=image_folder, texture_enum=texture_type
    )
    if run_vis:
        mesh.vis_pv()
    if run_aggregation:
        mesh.aggregate_viepoints_pytorch3d()
        # mesh.aggregate_viewpoints_naive()
    if run_render:
        mesh.render_pytorch3d()


if __name__ == "__main__":
    args = parse_args()
    main(
        args.mesh_file,
        args.camera_file,
        args.image_folder,
        args.run_vis,
        args.run_aggregation,
        args.run_render,
        args.texture_type,
    )
