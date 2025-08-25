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
import typing
from pathlib import Path

import numpy as np
import pyproj
import pyvista as pv
from matplotlib.pyplot import Normalize, cm

from geograypher.cameras.derived_cameras import MetashapeCameraSet
from geograypher.constants import PATH_TYPE
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
        "--camera-file",
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
        "--mesh-crs",
        type=int,
        required=True,
        help="The CRS to interpret the mesh in.",
    )
    parser.add_argument(
        "--original-image-folder",
        type=Path,
        required=False,
        help="If provided, this will be subtracted off the beginning of absolute image paths"
        " stored in the camera_file. See MetashapeCameraSet for details.",
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
        help="Height threshold (same units as DTM → m) for ground/aboveground separation."
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
        args.camera_file.exists()
    ), f"Camera XML file does not exist: {args.camera_file}"
    assert args.dtm_file.exists(), f"DTM file does not exist: {args.dtm_file}"
    assert args.mesh_file.exists(), f"Mesh file does not exist: {args.mesh_file}"

    # Ensure output folder exists
    ensure_folder(args.output_folder)
    # Ensure vis folder exists if provided
    if args.vis_folder is not None:
        ensure_folder(args.vis_folder)

    return args


def render_height_masks(
    image_folder: PATH_TYPE,
    camera_file: PATH_TYPE,
    mesh_file: PATH_TYPE,
    dtm_file: PATH_TYPE,
    mesh_CRS: pyproj.CRS,
    original_image_folder: typing.Optional[PATH_TYPE],
    output_folder: PATH_TYPE,
    output_mode: str,
    threshold_cutoff: float,
    vis_folder: typing.Optional[PATH_TYPE],
    vis_n_images: int,
):
    """
    Render height masks made from a mesh and digital terrain model (DTM), saving renders
    geofrom each camera's perspective.

    Arguments:
        image_folder: PATH_TYPE, Path to the folder containing images matching the camera
            XML data or a subset thereof. This script will iterate over available images
            in this folder.
        camera_file: PATH_TYPE, Path to the Metashape camera XML file with camera positions.
        mesh_file: PATH_TYPE, Path to the mesh file (e.g., .ply) that we will assess point
            height on.
        dtm_file: PATH_TYPE, Path to the digital terrain model (DTM) raster file (usually a tif).
        mesh_CRS: pyproj.CRS, the CRS to interpret the mesh in.
        original_image_folder: typing.Optional[PATH_TYPE], If provided, this will be subtracted
            off the beginning of absolute image paths stored in the camera_file. See
            MetashapeCameraSet for details.
        output_folder: PATH_TYPE, Folder to save the rendered ground masks. Will be created
            if it doesn't exist
        output_mode: str, How to render the output: 'threshold' will render scenes with values
            0=invalid, 1=below cutoff, 2=above cutoff; 'raw' will render scenes with the
            raw height values from the camera perspective.
        threshold_cutoff: float, Height threshold (same units as DTM → m) for ground/aboveground
            separation. Only used if output_mode is 'threshold'.
        vis_folder: typing.Optional[PATH_TYPE], If provided, a textured mesh and evaluation
            images will be saved in this folder.
        vis_n_images: int, Number of eval images to save in the vis folder (only used if
            vis_folder given).

    Raises:
        ValueError: the output mode is an invalid mode
    """

    def load_mesh(texture=None):
        """Small helper function for something we repeat."""
        return TexturedPhotogrammetryMesh(
            mesh_file,
            input_CRS=mesh_CRS,
            texture=texture,
        )

    mesh = load_mesh()

    # Calculate the height of each mesh vertex above the detected ground (DTM).
    # Note that DTM files usually cover a smaller area spatially than the mesh,
    # and points outside the DTM ROI will have a height of NaN.
    height = mesh.get_height_above_ground(DTM_file=dtm_file)

    if output_mode == "threshold":
        # Sort the ground height into mesh texture, so that 0=nan, 1=ground, 2=aboveground
        texture = np.zeros(len(height), dtype=float)
        texture[np.isnan(height)] = 0
        texture[(~np.isnan(height)) & (height <= threshold_cutoff)] = 1
        texture[(~np.isnan(height)) & (height > threshold_cutoff)] = 2
        cast_to_uint8 = True
        label_suffix = ".tif"
        save_as_npy = False
    elif output_mode == "raw":
        # Just use the raw height values to retexture the mesh
        texture = height
        cast_to_uint8 = False
        label_suffix = ".npy"
        save_as_npy = True
    else:
        raise ValueError(f"Unknown mode: {output_mode}")

    # Reload the same mesh, but applying the height-labeled texture to the vertices
    height_mesh = load_mesh(texture=texture.reshape(-1, 1))

    # Load camera metadata
    camera_set = MetashapeCameraSet(
        camera_file,
        image_folder,
        original_image_folder=original_image_folder,
        validate_images=True,
    )
    extension = Path(camera_set.cameras[0].get_image_filename()).suffix

    if vis_folder is not None:
        # Save an evaluation mesh
        if output_mode == "threshold":
            # Note that we have to divide by 10 to get (0, 1, 2) to fit nicely into
            # the 0-1 range of tab10
            colored = cm.get_cmap("tab10")(texture.flatten() / 10)
        else:
            normalize = Normalize(vmin=np.nanmin(texture), vmax=np.nanmax(texture))
            colored = cm.get_cmap("viridis")(normalize(texture))
        vis_mesh = load_mesh(texture=(colored[:, :3] * 255).astype(np.uint8))
        vis_mesh.save_mesh(vis_folder / "height_mesh.ply", save_vert_texture=True)

    # For each camera, render the height-painted mesh onto that camera view
    height_mesh.save_renders(
        camera_set,
        output_folder=output_folder,
        save_native_resolution=True,
        cast_to_uint8=cast_to_uint8,
        save_as_npy=save_as_npy,
    )

    if vis_folder is not None:
        # Save evaluation images of the ground masks
        show_segmentation_labels(
            label_folder=output_folder,
            image_folder=image_folder,
            savefolder=vis_folder,
            num_show=vis_n_images,
            image_suffix=extension,
            label_suffix=label_suffix,
        )


if __name__ == "__main__":
    args = parse_args()
    render_height_masks(
        image_folder=args.image_folder,
        camera_file=args.camera_file,
        mesh_file=args.mesh_file,
        dtm_file=args.dtm_file,
        mesh_CRS=args.mesh_crs,
        original_image_folder=args.original_image_folder,
        output_folder=args.output_folder,
        output_mode=args.output_mode,
        threshold_cutoff=args.threshold_cutoff,
        vis_folder=args.vis_folder,
        vis_n_images=args.vis_n_images,
    )
