"""
Render height masks made from a mesh and digital terrain model (DTM), saving renders
from each camera's perspective. There are two options:
- Threshold - Render a discrete mask with below ground, above ground, and invalid points.
    Determining which points are below vs. above ground is done by checking the height
    of mesh vertices above the DTM against a threshold.
- Raw - Render the mesh from each camera view, where the pixel value is the height of
    the mesh at that point above the STM. The output is an (M, N) numpy array instead
    of an image file.
"""

import argparse
from pathlib import Path

import numpy as np
import pyvista as pv
from matplotlib.pyplot import Normalize, cm

from geograypher.cameras.derived_cameras import MetashapeCameraSet
from geograypher.meshes.meshes import TexturedPhotogrammetryMesh
from geograypher.utils.files import ensure_folder
from geograypher.utils.visualization import show_segmentation_labels


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--image-folder",
        type=Path,
        required=True,
        help="Path to the folder containing images matching the camera XML data"
        " or a subset thereof. This script will iterate over available images"
        " in this folder.",
    )
    parser.add_argument(
        "--camera-xml",
        type=Path,
        required=True,
        help="Path to the Metashape camera XML file with camera positions.",
    )
    parser.add_argument(
        "--mesh-file",
        type=Path,
        required=True,
        help="Path to the mesh file (e.g., .ply) that we will assess point height on.",
    )
    parser.add_argument(
        "--dtm-file",
        type=Path,
        required=True,
        help="Path to the digital terrain model (DTM) raster file (usually a tif).",
    )
    parser.add_argument(
        "--output-folder",
        type=Path,
        required=True,
        help="Folder to save the rendered ground masks. Will be created if it doesn't exist",
    )
    parser.add_argument(
        "--output-mode",
        type=str,
        choices=["threshold", "raw"],
        default="raw",
        help="How to render the output: 'threshold' will render scenes with values"
        " 0=invalid, 1=below cutoff, 2=above cutoff; 'raw' will render scenes with the"
        " raw height values from the camera perspective.",
    )
    parser.add_argument(
        "--threshold-cutoff",
        type=float,
        default=1.0,
        help="Height threshold (in same units as DTM) for ground/aboveground separation."
        " Only used if --output-mode is 'threshold'.",
    )
    parser.add_argument(
        "--vis-folder",
        type=Path,
        required=False,
        help="If provided, a textured mesh and evaluation images will be saved in this folder.",
    )
    parser.add_argument(
        "--vis-n-images",
        type=int,
        default=10,
        help="Number of eval images to save in the vis folder (only used if --vis-folder given).",
    )
    args = parser.parse_args()

    # Assert existence of all required Path arguments except output_folders
    assert (
        args.image_folder.exists()
    ), f"Image folder does not exist: {args.image_folder}"
    assert (
        args.camera_xml.exists()
    ), f"Camera XML file does not exist: {args.camera_xml}"
    assert args.dtm_file.exists(), f"DTM file does not exist: {args.dtm_file}"
    assert args.mesh_file.exists(), f"Mesh file does not exist: {args.mesh_file}"

    # Ensure output folder exists
    ensure_folder(args.output_folder)
    # Ensure vis folder exists if provided
    if args.vis_folder is not None:
        ensure_folder(args.vis_folder)

    return args


def main():
    args = parse_args()

    def load_mesh(texture=None):
        """Small helper function for something we repeat."""
        return TexturedPhotogrammetryMesh(
            args.mesh_file,
            transform_filename=args.camera_xml,
            require_transform=True,
            texture=texture,
        )

    mesh = load_mesh()

    # Calculate the height of each mesh vertex above the detected ground (DTM)
    height = mesh.get_height_above_ground(DTM_file=args.dtm_file)

    if args.output_mode == "threshold":
        # Sort the ground height into mesh texture, so that 0=nan, 1=ground, 2=aboveground
        texture = np.zeros(len(height), dtype=float)
        texture[np.isnan(height)] = 0
        texture[(~np.isnan(height)) & (height <= args.threshold_cutoff)] = 1
        texture[(~np.isnan(height)) & (height > args.threshold_cutoff)] = 2
        cast_to_uint8 = True
        label_suffix = ".png"
    elif args.output_mode == "raw":
        # Just use the raw height values to retexture the mesh
        texture = height
        cast_to_uint8 = False
        label_suffix = ".npy"
    else:
        raise NotImplementedError(f"Unknown mode: {args.output_mode}")

    # Reload the same mesh, but applying the height-labeled texture to the vertices
    height_mesh = load_mesh(texture=texture.reshape(-1, 1))

    # Load camera metadata
    camera_set = MetashapeCameraSet(args.camera_xml, args.image_folder)

    # If your image folder has a subset of images, use those image
    # names to limit your renders to only the matching camera subset
    extension = Path(camera_set.cameras[0].get_image_filename()).suffix
    imset = set([im.name for im in args.image_folder.glob(f"*{extension}")])
    assert len(imset) > 0, f"No images found in {args.image_folder} with *{extension}"
    # Limit the cameras to a subset, potentially
    camera_set.cameras = [
        cam
        for cam in camera_set.cameras
        if Path(cam.get_image_filename()).name in imset
    ]

    if args.vis_folder is not None:
        # Save an evaluation mesh
        if args.output_mode == "threshold":
            cmap = np.array(
                [[170, 0, 0], [140, 140, 255], [90, 200, 90]], dtype=np.uint8
            )
            colored = cmap[texture.flatten().astype(int)]
        else:
            normalize = Normalize(vmin=np.nanmin(texture), vmax=np.nanmax(texture))
            colored = (cm.get_cmap("viridis")(normalize(texture))[:, :3] * 255).astype(
                np.uint8
            )
        vis_mesh = load_mesh(texture=colored)
        vis_mesh.save_mesh(args.vis_folder / "height_mesh.ply", save_vert_texture=True)

    # For each camera, render the height-painted mesh onto that camera view
    height_mesh.save_renders(
        camera_set,
        output_folder=args.output_folder,
        save_native_resolution=True,
        cast_to_uint8=cast_to_uint8,
    )

    if args.vis_folder is not None:
        # Save evaluation images of the ground masks
        show_segmentation_labels(
            label_folder=args.output_folder,
            image_folder=args.image_folder,
            savefolder=args.vis_folder,
            num_show=args.vis_n_images,
            image_suffix=extension,
            label_suffix=label_suffix,
        )


if __name__ == "__main__":
    main()
