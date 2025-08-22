import argparse
import typing
from pathlib import Path

import fiona
import geopandas as gpd
import numpy as np
import pyproj
import shapely

from geograypher.cameras import MetashapeCameraSet
from geograypher.constants import PATH_TYPE
from geograypher.meshes import TexturedPhotogrammetryMesh
from geograypher.meshes.derived_meshes import TexturedPhotogrammetryMeshChunked
from geograypher.utils.visualization import show_segmentation_labels


# TODO Consider adding a mesh screenshot option or a vis savefolder
def render_labels(
    mesh_file: PATH_TYPE,
    cameras_file: PATH_TYPE,
    image_folder: PATH_TYPE,
    texture: typing.Union[PATH_TYPE, np.ndarray, None],
    render_savefolder: PATH_TYPE,
    mesh_CRS: pyproj.CRS,
    original_image_folder: typing.Union[PATH_TYPE, None] = None,
    subset_images_savefolder: typing.Union[PATH_TYPE, None] = None,
    texture_column_name: typing.Union[str, None] = None,
    DTM_file: typing.Union[PATH_TYPE, None] = None,
    ground_height_threshold: typing.Union[float, None] = None,
    render_ground_class: bool = False,
    textured_mesh_savefile: typing.Union[PATH_TYPE, None] = None,
    ROI: typing.Union[PATH_TYPE, gpd.GeoDataFrame, shapely.MultiPolygon, None] = None,
    mesh_ROI_buffer_radius_meters: float = 50,
    cameras_ROI_buffer_radius_meters: float = 150,
    IDs_to_labels: typing.Union[dict, None] = None,
    render_image_scale: float = 1,
    mesh_downsample: float = 1,
    n_render_clusters: typing.Union[int, None] = None,
    cast_to_uint8: bool = True,
    save_as_npy: bool = False,
    vis: bool = False,
    mesh_vis_file: typing.Union[PATH_TYPE, None] = None,
    labels_vis_folder: typing.Union[PATH_TYPE, None] = None,
):
    """Renders image-based labels using geospatial ground truth data

    Args:
        mesh_file (PATH_TYPE):
            Path to the Metashape-exported mesh file
        cameras_file (PATH_TYPE):
            Path to the MetaShape-exported .xml cameras file
        image_folder (PATH_TYPE):
            Path to the folder of images used to create the mesh
        texture (typing.Union[PATH_TYPE, np.ndarray, None]):
            See TexturedPhotogrammetryMesh.load_texture
        render_savefolder (PATH_TYPE):
            Where to save the rendered labels
        mesh_CRS: (pyproj.CRS):
            The vertex coordinates of the input mesh should be interpreteted in this coordinate
            references system to georeference them. Since meshes are not commonly used for
            geospatial tasks, there isn't a common standard for encoding this information in the mesh.
        original_image_folder (typing.Union[PATH_TYPE, None], optional):
            Where the images were when photogrammetry was run. Metashape saves imagenames with an
            absolute path which can cause issues. If this argument is provided, this path is removed
            from the start of each image file name, which allows the camera set to be used with a
            moved folder of images specified by `image_folder`. Defaults to None.
        subset_images_savefolder (typing.Union[PATH_TYPE, None], optional):
            Where to save the subset of images for which labels are generated. Defaults to None.
        texture_column_name (typing.Union[str, None], optional):
            Column to use in vector file for texture information". Defaults to None.
        DTM_file (typing.Union[PATH_TYPE, None], optional):
            Path to a DTM file to use for ground thresholding. Defaults to None.
        ground_height_threshold (typing.Union[float, None], optional):
            Set points under this height to ground. Only applicable if DTM_file is provided. Defaults to None.
        render_ground_class (bool, optional):
            Should the ground class be included in the renders or deleted.. Defaults to False.
        textured_mesh_savefile (typing.Union[PATH_TYPE, None], optional):
            Where to save the textured and subsetted mesh, if needed in the future. Defaults to None.
        ROI (typing.Union[PATH_TYPE, gpd.GeoDataFrame, shapely.MultiPolygon, None], optional):
            The region of interest to render labels for. Defaults to None.
        mesh_ROI_buffer_radius_meters (float, optional):
            The distance in meters to include around the ROI for the mesh. Defaults to 50.
        cameras_ROI_buffer_radius_meters (float, optional):
            The distance in meters to include around the ROI for the cameras. Defaults to 150.
        IDs_to_labels (typing.Union[None, dict], optional):
            Mapping between the integer labels and string values for the classes. Defaults to None.
        render_image_scale (float, optional):
            Downsample the images to this fraction of the size for increased performance but lower quality. Defaults to 1.
        mesh_downsample (float, optional):
            Downsample the mesh to this fraction of vertices for increased performance but lower quality. Defaults to 1.
        n_render_clusters (typing.Union[int, None]):
            If set, break the camera set and mesh into this many clusters before rendering. This is
            useful for large meshes that are otherwise very slow. Defaults to None.
        cast_to_uint8 (bool, optional):
            If True, cast the rendered labels to uint8. If False, it is cast to uint16 or uint32, if not saved as .npy.
            Defaults to True.
        save_as_npy (bool, optional):
            If True, save the rendered labels are float64 and saved as numpy arrays. Defaults to False.
        mesh_vis (typing.Union[PATH_TYPE, None])
            Path to save the visualized mesh instead of showing it interactively. Only applicable if vis=True. Defaults to None.
        labels_vis (typing.Union[PATH_TYPE, None])
            Defaults to None.
    """
    ## Determine the ROI
    # If the ROI is unset and the texture is a geodataframe, set the ROI to that
    if ROI is None and isinstance(texture, gpd.GeoDataFrame):
        ROI = texture
    elif ROI is None and isinstance(texture, (str, Path)):
        try:
            ROI = gpd.read_file(texture)
        except fiona.errors.DriverError:
            pass

    ## Create the camera set
    # This is done first because it's often faster than mesh operations which
    # makes it a good place to check for failures
    camera_set = MetashapeCameraSet(
        cameras_file, image_folder, original_image_folder=original_image_folder
    )

    if ROI is not None:
        # Extract cameras near the training data
        camera_set = camera_set.get_subset_ROI(
            ROI=ROI, buffer_radius=cameras_ROI_buffer_radius_meters, is_geospatial=True
        )
    # If requested, save out the images corresponding to this subset of cameras.
    # This is useful for model training.
    if subset_images_savefolder is not None:
        camera_set.save_images(subset_images_savefolder)

    # Select whether to use a class that renders by chunks or not
    MeshClass = (
        TexturedPhotogrammetryMesh
        if n_render_clusters is None
        else TexturedPhotogrammetryMeshChunked
    )

    ## Create the textured mesh
    mesh = MeshClass(
        mesh_file,
        input_CRS=mesh_CRS,
        downsample_target=mesh_downsample,
        texture=texture,
        texture_column_name=texture_column_name,
        ROI=ROI,
        ROI_buffer_meters=mesh_ROI_buffer_radius_meters,
        IDs_to_labels=IDs_to_labels,
    )

    ## Set the ground class if applicable
    if DTM_file is not None and ground_height_threshold is not None:
        # The ground ID will be set to the next value if None, or np.nan if np.nan
        ground_ID = None if render_ground_class else np.nan
        mesh.label_ground_class(
            DTM_file=DTM_file,
            height_above_ground_threshold=ground_height_threshold,
            only_label_existing_labels=True,
            ground_class_name="GROUND",
            ground_ID=ground_ID,
            set_mesh_texture=True,
        )

    # Save the textured and subsetted mesh, if applicable
    if textured_mesh_savefile is not None:
        mesh.save_mesh(textured_mesh_savefile)

    # Show the cameras and mesh if requested
    if vis or mesh_vis_file is not None:
        mesh.vis(camera_set=camera_set, screenshot_filename=mesh_vis_file)

    # Include n_render_clusters as an optional keyword argument, if provided. This is only applicable
    # if this mesh is a TexturedPhotogrammetryMeshChunked object
    render_kwargs = (
        {} if n_render_clusters is None else {"n_clusters": n_render_clusters}
    )
    # Render the labels and save them. This is the slow step.
    mesh.save_renders(
        camera_set=camera_set,
        render_image_scale=render_image_scale,
        save_native_resolution=True,
        output_folder=render_savefolder,
        make_composites=False,
        cast_to_uint8=cast_to_uint8,
        save_as_npy=save_as_npy,
        **render_kwargs,
    )

    if vis or labels_vis_folder is not None:
        # Show some examples of the rendered labels side-by-side with the real images
        show_segmentation_labels(
            label_folder=render_savefolder,
            image_folder=image_folder,
            savefolder=labels_vis_folder,
            label_suffix=".tif",
            num_show=10,
        )


def parse_args():
    """Parse and return arguements

    Returns:
        argparse.Namespace: Arguments
    """
    description = (
        "This script renders labels onto individual images using geospatial textures. "
        + "By default is uses the example data. All arguments are passed to "
        + "geograypher.entrypoints.workflow_functions.render_labels "
        + "which has the following documentation:\n\n"
        + render_labels.__doc__
    )
    # Ideally we'd include the defaults for each argument, but there is no help text so the
    # ArgumentDefaultsHelpFormatter formatter doesn't show them
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Add arguments
    parser.add_argument("--mesh-file", type=Path, required=True)
    parser.add_argument("--mesh-CRS", required=True)
    parser.add_argument("--cameras-file", type=Path, required=True)
    parser.add_argument("--image-folder", type=Path, required=True)
    parser.add_argument("--texture", type=Path, required=True)
    parser.add_argument("--render-savefolder", type=Path, required=True)
    parser.add_argument("--original-image-folder", type=Path)
    parser.add_argument("--subset-images-savefolder", type=Path)
    parser.add_argument("--texture-column-name")
    parser.add_argument("--DTM-file")
    parser.add_argument("--ground-height-threshold", type=float, default=2.0)
    parser.add_argument("--render-ground-class", action="store_true")
    parser.add_argument("--textured-mesh-savefile")
    parser.add_argument("--ROI")
    parser.add_argument("--mesh-ROI-buffer-radius-meters", default=50, type=float)
    parser.add_argument("--cameras-ROI-buffer-radius-meters", default=100, type=float)
    parser.add_argument("--render-image-scale", type=float, default=1)
    parser.add_argument("--mesh-downsample", type=float, default=1)
    parser.add_argument("--vis", action="store_true")
    parser.add_argument("--mesh-vis-file", type=Path)
    parser.add_argument("--labels-vis-folder", type=Path)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # Parse command line args
    args = parse_args()
    # Pass all the arguments command line options to render_labels
    render_labels(**args.__dict__)
