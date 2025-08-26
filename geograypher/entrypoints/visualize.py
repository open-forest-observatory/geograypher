import argparse
import typing

import numpy as np
import pyproj

from geograypher.cameras import MetashapeCameraSet
from geograypher.constants import PATH_TYPE
from geograypher.meshes import TexturedPhotogrammetryMesh
from geograypher.utils.indexing import find_argmax_nonzero_value


def visualize(
    mesh_file: PATH_TYPE,
    mesh_CRS: typing.Union[pyproj.CRS, int],
    camera_file: PATH_TYPE,
    texture: typing.Optional[PATH_TYPE] = None,
    texture_column_name: typing.Optional[str] = None,
    IDs_to_labels: typing.Union[PATH_TYPE, dict, None] = None,
    downsample_target: float = 1.0,
    ROI: typing.Optional[PATH_TYPE] = None,
    ROI_buffer_meters: float = 0.0,
    convert_texture_to_max_class: bool = False,
):
    """
    Utility for visualizing meshes and associated camera sets. Note that this should be run in a
    graphical desktop.


    Args:
        mesh_file (PATH_TYPE):
            Path to the mesh file (often .ply)
        mesh_CRS (typing.Union[pyproj.CRS, int]):
            The vertex coordinates of the input mesh should be interpreteted in this coordinate
            references system to georeference them. Since meshes are not commonly used for
            geospatial tasks, there isn't a common standard for encoding this information in the mesh.
        camera_file (PATH_TYPE):
            Path to the MetaShape-exported .xml cameras file
        texture (typing.Union[PATH_TYPE, None]):
            See TexturedPhotogrammetryMesh.load_texture
        texture_column_name (typing.Union[str, None], optional):
            Column to use in vector file for texture information". Defaults to None.
        IDs_to_labels (typing.Union[PATH_TYPE, dict, None], optional):
            Mapping between the integer labels and string values for the classes or a path to a
            .json file representing the same. Defaults to None.
        downsample_target (float):
            Downsample the mesh to this fraction of vertices for increased performance but lower
            quality. Defaults to 1.
        ROI (PATH, optional):
            The region of interest to render labels for. Defaults to None.
        ROI_buffer_meters (typing.Optional[float]):
            The distance in meters to include around the ROI for the mesh. Defaults to 0.0.
        convert_texture_to_max_class (bool, optional):
            If the texture is a path to an .npy file encoding a (n_faces, n_classes) matrix of
            per-class weights, convert this representation into an (n_faces,) representation
            indicating the most common non-zero class. This is helpful for displaying the results
            of aggregated classification predictions. Defaults to False.
    """

    if convert_texture_to_max_class:
        # Load the texture
        texture = np.load(texture)
        # Convert to the argmax representation with nans for rows where all values were zero
        texture = find_argmax_nonzero_value(texture)

    # Load the camera set if provided
    if camera_file is not None:
        camera_set = MetashapeCameraSet(camera_file=camera_file, image_folder="")
    else:
        camera_set = None

    # If the camera set is provided and the ROI is not None, subset the cameras to that ROI
    if camera_set is not None and ROI is not None:
        camera_set = camera_set.get_subset_ROI(ROI, buffer_radius=ROI_buffer_meters)

    # Create the mesh
    mesh = TexturedPhotogrammetryMesh(
        mesh_file,
        input_CRS=mesh_CRS,
        ROI=ROI,
        ROI_buffer_meters=ROI_buffer_meters,
        downsample_target=downsample_target,
        texture=texture,
        texture_column_name=texture_column_name,
        IDs_to_labels=IDs_to_labels,
    )

    # Visualize the mesh and the camera set if provided
    # TODO optional visualization arguments like the colormap could be provided here
    mesh.vis(camera_set=camera_set)


def parse_args():
    """Parse and return arguements

    Returns:
        argparse.Namespace: Arguments
    """
    description = (
        "This script visualizes a mesh, optionally with added textures or a camera set. "
        + " All arguments are passed to geograypher.entrypoints.visualize.visualize "
        + "which has the following documentation:\n\n"
        + visualize.__doc__
    )
    # Ideally we'd include the defaults for each argument, but there is no help text so the
    # ArgumentDefaultsHelpFormatter formatter doesn't show them
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh-file", required=True)
    parser.add_argument("--mesh-CRS", required=True)
    parser.add_argument("--camera-file", required=True)
    parser.add_argument("--texture")
    parser.add_argument("--texture-column-name")
    parser.add_argument(
        "--IDs-to-labels",
        help="When using the CLI, path to a .json file must be provided, not a dict representation",
    )
    parser.add_argument("--downsample-target", type=float, default=1.0)
    parser.add_argument("--ROI")
    parser.add_argument("--ROI-buffer-meters", type=float, default=0.0)
    parser.add_argument("--convert-texture-to-max-class", action="store_true")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    visualize(**args.__dict__)
