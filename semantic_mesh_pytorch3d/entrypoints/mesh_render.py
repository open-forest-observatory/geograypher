from semantic_mesh_pytorch3d.config import DATA_FOLDER
from semantic_mesh_pytorch3d.meshes import Pytorch3DMesh
from argparse import ArgumentParser
from pathlib import Path


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--mesh-file",
        default=str(
            Path(
                DATA_FOLDER,
                "9-image-emerald-point-reconstruction",
                "9-image-reconstruction.ply",
            )
        ),
    )
    parser.add_argument(
        "--camera-file",
        default=str(
            Path(
                DATA_FOLDER,
                "9-image-emerald-point-reconstruction",
                "9-image-reconstruction",
            )
        ),
    )
    parser.add_argument(
        "--image-folder",
        default=str(
            Path(DATA_FOLDER, "9-image-emerald-point-reconstruction", "images")
        ),
    )
    args = parser.parse_args()
    return args


def main(mesh_file, camera_file, image_folder):
    mesh = Pytorch3DMesh(mesh_file, camera_file, image_folder=image_folder)
    mesh.vis_pv()
    mesh.render()
    mesh.render_geometric()


if __name__ == "__main__":
    args = parse_args()
    main(args.mesh_file, args.camera_file, args.image_folder)
