"""
Render ground/aboveground masks made from a mesh and digital terrain model (DTM), saving renders
from each camera's perspective. Determining which points are below vs. above ground is done
by checking the height of mesh vertices above the DTM.
"""

import argparse
import numpy as np
import pyvista as pv
from pathlib import Path

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
        help="Path to the folder containing images matching the camera XML data.",
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
        "--cutoff",
        type=float,
        default=1.0,
        help="Height threshold (in same units as DTM) for ground/aboveground separation.",
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
    assert args.image_folder.exists(), f"Image folder does not exist: {args.image_folder}"
    assert args.camera_xml.exists(), f"Camera XML file does not exist: {args.camera_xml}"
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

    # Sort the ground height into texture, so that 0=nan, 1=ground, 2=aboveground
    ground_mask = mesh.get_height_above_ground(DTM_file=args.dtm_file)
    texture = np.zeros(len(ground_mask), dtype=float)
    texture[ground_mask > args.cutoff] = 2
    texture[ground_mask <= args.cutoff] = 1
    texture[np.isnan(ground_mask)] = 0

    # Reload the same mesh, but applying the height-labeled texture to the vertices
    ground_mesh = load_mesh(texture=texture.reshape(-1, 1))

    # Load camera metadata
    camera_set = MetashapeCameraSet(args.camera_xml, args.image_folder)

    # If your image folder has a subset of images, use those image
    # names to limit your renders to only the matching camera subset
    extension = Path(camera_set.cameras[0].get_image_filename()).suffix
    imset = set([im.name for im in args.image_folder.glob(f"*{extension}")])
    assert len(imset) > 0, f"No images found in {args.image_folder} with *{extension}"
    # Limit the cameras to a subset, potentially
    camera_set.cameras = [
        cam for cam in camera_set.cameras
        if Path(cam.get_image_filename()).name in imset
    ]

    # Note that the render_flat call in save_renders removes vertex textures. That
    # means we need to save an evaluation mesh beforehand, if relevant
    if args.vis_folder is not None:
        # Change the 0, 1, 2 textures to a color mapped version and save the mesh
        cmap = np.array([[170, 0, 0], [140, 140, 255], [90, 200, 90]], dtype=np.uint8)
        colored = cmap[ground_mesh.get_texture(request_vertex_texture=True).flatten().astype(int)]
        vis_mesh = load_mesh(texture=colored)
        vis_mesh.save_mesh(args.vis_folder / "ground_mesh.ply", save_vert_texture=True)

    # For each camera, render the height-painted mesh onto that camera view
    ground_mesh.save_renders(
        camera_set,
        output_folder=args.output_folder,
        save_native_resolution=True,
    )

    if args.vis_folder is not None:
        # Save evaluation images of the ground masks
        show_segmentation_labels(
            label_folder=args.output_folder,
            image_folder=args.image_folder,
            savefolder=args.vis_folder,
            num_show=args.vis_n_images,
            image_suffix=extension,
        )


if __name__ == "__main__":
    main()
