from argparse import ArgumentParser
from pathlib import Path

from semantic_mesh_pytorch3d.config import DATA_FOLDER
from semantic_mesh_pytorch3d.meshes import Pytorch3DMesh


def parse_args():
    parser = ArgumentParser()
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
                "2023-08-23_1027_QR_F1_rgb_100m_25_img_20230920T1527_cameras",
            )
        ),
    )
    parser.add_argument(
        "--image-folder",
        default=str(Path(DATA_FOLDER, "2023-08-23_1027_QR_F1_rgb_100m_25_img", "images")),
    )
    args = parser.parse_args()
    return args


def main(mesh_file, camera_file, image_folder):
    mesh = Pytorch3DMesh(mesh_file, camera_file, image_folder=image_folder)
    mesh.vis_pv()
    mesh.aggregate_numpy()
    mesh.render_pytorch3d()


if __name__ == "__main__":
    args = parse_args()
    main(args.mesh_file, args.camera_file, args.image_folder)
