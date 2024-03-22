import numpy as np
import typing
import geopandas as gpd
import shapely
from pathlib import Path


from geograypher.meshes import TexturedPhotogrammetryMesh
from geograypher.cameras import MetashapeCameraSet
from geograypher.constants import PATH_TYPE
from geograypher.utils.visualization import show_segmentation_labels


# TODO Consider adding a mesh screenshot option or a vis savefolder
def render_labels(
    mesh_file: PATH_TYPE,
    cameras_file: PATH_TYPE,
    image_folder: PATH_TYPE,
    texture: typing.Union[PATH_TYPE, np.ndarray, None],
    render_savefolder: PATH_TYPE,
    transform_file: typing.Union[PATH_TYPE, None] = None,
    subset_images_savefolder: typing.Union[PATH_TYPE, None] = None,
    texture_column_name: typing.Union[str, None] = None,
    DTM_file: typing.Union[PATH_TYPE, None] = None,
    ground_height_threshold: typing.Union[float, None] = None,
    render_ground_class: bool = False,
    textured_mesh_savefile: typing.Union[PATH_TYPE, None] = None,
    ROI: typing.Union[PATH_TYPE, gpd.GeoDataFrame, shapely.MultiPolygon, None] = None,
    ROI_buffer_radius_meters: float = 50,
    render_image_scale: float = 1,
    mesh_downsample: float = 1,
    vis: bool = False,
):
    ## Determine the ROI
    # If the ROI is unset and the texture is a spatial file, set the ROI to that
    if ROI is None and isinstance(texture, (str, Path, gpd.GeoDataFrame)):
        ROI = texture

    # If the transform filename is None, use the cameras filename instead
    # since this contains the transform information
    if transform_file is None:
        transform_file = cameras_file

    ## Create the camera set
    # This is done first because it's often faster than mesh operations which
    # makes it a good place to check for failures
    camera_set = MetashapeCameraSet(cameras_file, image_folder)
    # Extract cameras near the training data
    training_camera_set = camera_set.get_subset_ROI(
        ROI=ROI, buffer_radius_meters=ROI_buffer_radius_meters
    )
    # If requested, save out the images corresponding to this subset of cameras.
    # This is useful for model training.
    if subset_images_savefolder is not None:
        training_camera_set.save_images(subset_images_savefolder)

    ## Create the textured mesh
    mesh = TexturedPhotogrammetryMesh(
        mesh_file,
        downsample_target=mesh_downsample,
        texture=texture,
        texture_column_name=texture_column_name,
        transform_filename=transform_file,
        ROI=ROI,
        ROI_buffer_meters=ROI_buffer_radius_meters,
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
    if vis:
        mesh.vis(camera_set=training_camera_set)

    # Render the labels and save them. This is the slow step.
    mesh.save_renders_pytorch3d(
        camera_set=training_camera_set,
        render_image_scale=render_image_scale,
        save_native_resolution=True,
        output_folder=render_savefolder,
    )

    if vis:
        # Show some examples of the rendered labels side-by-side with the real images
        show_segmentation_labels(
            label_folder=render_savefolder,
            image_folder=image_folder,
            num_show=10,
        )
