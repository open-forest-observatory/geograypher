import typing
from pathlib import Path

import geopandas as gpd
import numpy as np
import shapely

from geograypher.cameras import MetashapeCameraSet
from geograypher.constants import PATH_TYPE
from geograypher.meshes import TexturedPhotogrammetryMesh
from geograypher.segmentation import SegmentorPhotogrammetryCameraSet
from geograypher.segmentation.derived_segmentors import LookUpSegmentor
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
        transform_file (typing.Union[PATH_TYPE, None], optional):
            File containing the transform from local coordinates to EPSG:4978. Defaults to None.
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
        ROI_buffer_radius_meters (float, optional):
            The distance in meters to include around the ROI. Defaults to 50.
        render_image_scale (float, optional):
            Downsample the images to this fraction of the size for increased performance but lower quality. Defaults to 1.
        mesh_downsample (float, optional):
            Downsample the mesh to this fraction of vertices for increased performance but lower quality. Defaults to 1.
        vis (bool, optional):
            Show mesh and rendered labels. Defaults to False.
    """
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


def aggregate_images(
    mesh_file: PATH_TYPE,
    cameras_file: PATH_TYPE,
    image_folder: PATH_TYPE,
    label_folder: PATH_TYPE,
    subset_images_folder: typing.Union[PATH_TYPE, None] = None,
    mesh_transform_file: typing.Union[PATH_TYPE, None] = None,
    DTM_file: typing.Union[PATH_TYPE, None] = None,
    height_above_ground_threshold: float = 2.0,
    ROI: typing.Union[PATH_TYPE, None] = None,
    ROI_buffer_radius_meters: float = 50,
    IDs_to_labels: typing.Union[dict, None] = None,
    mesh_downsample: float = 1.0,
    aggregate_image_scale: float = 1.0,
    aggregated_face_values_savefile: typing.Union[PATH_TYPE, None] = None,
    predicted_face_classes_savefile: typing.Union[PATH_TYPE, None] = None,
    top_down_vector_projection_savefile: typing.Union[PATH_TYPE, None] = None,
    vis: bool = False,
):
    """Aggregate labels from multiple viewpoints onto the surface of the mesh

    Args:
        mesh_file (PATH_TYPE): Path to the Metashape-exported mesh file
        cameras_file (PATH_TYPE): Path to the MetaShape-exported .xml cameras file
        image_folder (PATH_TYPE): Path to the folder of images used to create the mesh
        label_folder (PATH_TYPE): Path to the folder of labels to be aggregated onto the mesh. Must be in the same structure as the images
        subset_images_folder (typing.Union[PATH_TYPE, None], optional): Use only images from this subset. Defaults to None.
        mesh_transform_file (typing.Union[PATH_TYPE, None], optional): Transform from the mesh coordinates to the earth-centered, earth-fixed frame. Can be a 4x4 matrix represented as a .csv, or a Metashape cameras file containing the information. Defaults to None.
        DTM_file (typing.Union[PATH_TYPE, None], optional): Path to a digital terrain model file to remove ground points. Defaults to None.
        height_above_ground_threshold (float, optional): Height in meters above the DTM to consider ground. Only used if DTM_file is set. Defaults to 2.0.
        ROI (typing.Union[PATH_TYPE, None], optional): Geofile region of interest to crop the mesh to. Defaults to None.
        ROI_buffer_radius_meters (float, optional): Keep points within this distance of the provided ROI object, if unset, everything will be kept. Defaults to 50.
        IDs_to_labels (typing.Union[dict, None], optional): Maps from integer IDs to human-readable class name labels. Defaults to None.
        mesh_downsample (float, optional): Downsample the mesh to this fraction of vertices for increased performance but lower quality. Defaults to 1.0.
        aggregate_image_scale (float, optional): Downsample the labels before aggregation for faster runtime but lower quality. Defaults to 1.0.
        aggregated_face_values_savefile (typing.Union[PATH_TYPE, None], optional): Where to save the aggregated image values as a numpy array. Defaults to None.
        predicted_face_classes_savefile (typing.Union[PATH_TYPE, None], optional): Where to save the most common label per face texture as a numpy array. Defaults to None.
        top_down_vector_projection_savefile (typing.Union[PATH_TYPE, None], optional): Where to export the predicted map. Defaults to None.
        vis (bool, optional): Show the mesh model and predicted results. Defaults to False.
    """
    ## Create the camera set
    # Do the camera operations first because they are fast and good initial error checking
    camera_set = MetashapeCameraSet(cameras_file, image_folder)

    # If the ROI is not None, subset to cameras within a buffer distance of the ROI
    # TODO let get_subset_ROI accept a None ROI and return the full camera set
    if subset_images_folder is not None:
        camera_set = camera_set.get_cameras_in_folder(subset_images_folder)

    if ROI is not None and ROI_buffer_radius_meters is not None:
        # Extract cameras near the training data
        camera_set = camera_set.get_subset_ROI(
            ROI=ROI, buffer_radius_meters=ROI_buffer_radius_meters
        )

    if mesh_transform_file is None:
        mesh_transform_file = cameras_file

    ## Create the mesh
    mesh = TexturedPhotogrammetryMesh(
        mesh_file,
        transform_filename=mesh_transform_file,
        ROI=ROI,
        ROI_buffer_meters=ROI_buffer_radius_meters,
        IDs_to_labels=IDs_to_labels,
        downsample_target=mesh_downsample,
    )

    # Show the mesh if requested
    if vis:
        mesh.vis(camera_set=camera_set)

    # Create a segmentor object to load in the predictions
    segmentor = LookUpSegmentor(
        base_folder=image_folder,
        lookup_folder=label_folder,
        num_classes=np.max(list(mesh.get_IDs_to_labels().keys())) + 1,
    )
    # Create a camera set that returns the segmented images instead of the original ones
    segmentor_camera_set = SegmentorPhotogrammetryCameraSet(
        camera_set, segmentor=segmentor
    )

    ## Perform aggregation
    # this is the slow step
    aggregated_face_labels, _, _ = mesh.aggregate_viewpoints_pytorch3d(
        segmentor_camera_set,
        image_scale=aggregate_image_scale,
    )
    # If requested, save this data
    if aggregated_face_values_savefile is not None:
        Path(aggregated_face_values_savefile).parent.mkdir(exist_ok=True, parents=True)
        np.save(aggregated_face_values_savefile, aggregated_face_labels)

    # Find the most common class per face
    predicted_face_classes = np.argmax(
        aggregated_face_labels, axis=1, keepdims=True
    ).astype(float)

    # If requested, label the ground faces
    if DTM_file is not None and height_above_ground_threshold is not None:
        predicted_face_classes = mesh.label_ground_class(
            labels=predicted_face_classes,
            height_above_ground_threshold=height_above_ground_threshold,
            DTM_file=DTM_file,
            ground_ID=np.nan,
            set_mesh_texture=False,
        )

    if predicted_face_classes_savefile is not None:
        Path(predicted_face_classes_savefile).parent.mkdir(exist_ok=True, parents=True)
        np.save(predicted_face_classes_savefile, predicted_face_classes)

    if vis:
        # Show the mesh with predicted classes
        mesh.vis(vis_scalars=predicted_face_classes)

    # TODO this should be updated to take IDs_to_labels
    mesh.export_face_labels_vector(
        face_labels=np.squeeze(predicted_face_classes),
        export_file=top_down_vector_projection_savefile,
        vis=True,
    )
