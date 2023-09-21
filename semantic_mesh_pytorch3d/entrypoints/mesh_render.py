import argparse
from pathlib import Path

from semantic_mesh_pytorch3d.config import DATA_FOLDER
from semantic_mesh_pytorch3d.meshes import Pytorch3DMesh


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--mesh-file",
        default=str(
            Path(
                DATA_FOLDER,
                "2023-08-23_1027_QR_F1_rgb_100m_25_img",
                "exports",
                "2023-08-23_1027_QR_F1_rgb_100m_25_img_20230920T1527_model.ply",
            )
        ),
    )
    parser.add_argument(
        "--camera-file",
        default=str(
            Path(
                DATA_FOLDER,
                "2023-08-23_1027_QR_F1_rgb_100m_25_img",
                "exports",
                "2023-08-23_1027_QR_F1_rgb_100m_25_img_20230920T1527_cameras.xml",
            )
        ),
    )
    parser.add_argument(
        "--image-folder",
        default=str(
            Path(DATA_FOLDER, "2023-08-23_1027_QR_F1_rgb_100m_25_img", "images")
        ),
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


def main(mesh_file, camera_file, image_folder, run_vis, run_aggregation, run_render):
    mesh = Pytorch3DMesh(mesh_file, camera_file, image_folder=image_folder)
    if run_vis:
        mesh.vis_pv()
    if run_aggregation:
        mesh.aggregate_numpy()
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
    )
