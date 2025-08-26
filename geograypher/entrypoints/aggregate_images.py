import argparse
import json
import math
import typing
from pathlib import Path

import numpy as np
import pyproj

from geograypher.cameras import MetashapeCameraSet, SegmentorPhotogrammetryCameraSet
from geograypher.constants import EXAMPLE_IDS_TO_LABELS, PATH_TYPE
from geograypher.meshes import TexturedPhotogrammetryMesh
from geograypher.meshes.derived_meshes import TexturedPhotogrammetryMeshChunked
from geograypher.predictors.derived_segmentors import LookUpSegmentor
from geograypher.utils.files import ensure_containing_folder
from geograypher.utils.indexing import find_argmax_nonzero_value


def aggregate_images(
    mesh_file: PATH_TYPE,
    cameras_file: PATH_TYPE,
    image_folder: PATH_TYPE,
    label_folder: PATH_TYPE,
    mesh_CRS: pyproj.CRS,
    original_image_folder: typing.Union[PATH_TYPE, None] = None,
    subset_images_folder: typing.Union[PATH_TYPE, None] = None,
    filename_regex: typing.Optional[str] = None,
    take_every_nth_camera: typing.Union[int, None] = 100,
    DTM_file: typing.Union[PATH_TYPE, None] = None,
    height_above_ground_threshold: float = 2.0,
    ROI: typing.Union[PATH_TYPE, None] = None,
    ROI_buffer_radius_meters: float = 50,
    IDs_to_labels: typing.Union[dict, str, None] = None,
    mesh_downsample: float = 1.0,
    n_aggregation_clusters: typing.Union[int, None] = None,
    n_cameras_per_aggregation_cluster: typing.Union[int, None] = None,
    aggregate_image_scale: float = 1.0,
    aggregated_face_values_savefile: typing.Union[PATH_TYPE, None] = None,
    predicted_face_classes_savefile: typing.Union[PATH_TYPE, None] = None,
    top_down_vector_projection_savefile: typing.Union[PATH_TYPE, None] = None,
    vis: bool = False,
):
    """Aggregate labels from multiple viewpoints onto the surface of the mesh

    Args:
        mesh_file (PATH_TYPE):
            Path to the Metashape-exported mesh file
        cameras_file (PATH_TYPE):
            Path to the MetaShape-exported .xml cameras file
        image_folder (PATH_TYPE):
            Path to the folder of images used to create the mesh
        filename_regex (str, optional):
            Use only images with paths matching this regex
        label_folder (PATH_TYPE):
            Path to the folder of labels to be aggregated onto the mesh. Must be in the same
            structure as the images
        mesh_CRS (pyproj.CRS):
            The CRS to interpret the mesh in.
        original_image_folder (typing.Union[PATH_TYPE, None], optional):
            Where the images were when photogrammetry was run. Metashape saves imagenames with an
            absolute path which can cause issues. If this argument is provided, this path is removed
            from the start of each image file name, which allows the camera set to be used with a
            moved folder of images specified by `image_folder`. Defaults to None.
        subset_images_folder (typing.Union[PATH_TYPE, None], optional):
            Use only images from this subset. Defaults to None.
        take_every_nth_camera (typing.Union[int, None], optional):
            Downsample the camera set to only every nth camera if set. Defaults to None.
        DTM_file (typing.Union[PATH_TYPE, None], optional):
            Path to a digital terrain model file to remove ground points. Defaults to None.
        height_above_ground_threshold (float, optional):
            Height in meters above the DTM to consider ground. Only used if DTM_file is set.
            Defaults to 2.0.
        ROI (typing.Union[PATH_TYPE, None], optional):
            Geofile region of interest to crop the mesh to. Defaults to None.
        ROI_buffer_radius_meters (float, optional):
            Keep points within this distance of the provided ROI object, if unset, everything will
            be kept. Defaults to 50.
        IDs_to_labels (typing.Union[dict, None], optional):
            Maps from integer IDs to human-readable class name labels. Defaults to None.
        mesh_downsample (float, optional):
            Downsample the mesh to this fraction of vertices for increased performance but lower
            quality. Defaults to 1.0.
        n_aggregation_clusters (typing.Union[int, None]):
            If set, aggregate with this many clusters. Defaults to None.
        n_cameras_per_aggregation_cluster (typing.Union[int, None]):
            If set, and n_aggregation_clusters is not, use to compute a number of clusters such that
            each cluster has this many cameras. Defaults to None.
        aggregate_image_scale (float, optional):
            Downsample the labels before aggregation for faster runtime but lower quality. Defaults
            to 1.0.
        aggregated_face_values_savefile (typing.Union[PATH_TYPE, None], optional):
            Where to save the aggregated image values as a numpy array. Defaults to None.
        predicted_face_classes_savefile (typing.Union[PATH_TYPE, None], optional):
            Where to save the most common label per face texture as a numpy array. Defaults to None.
        top_down_vector_projection_savefile (typing.Union[PATH_TYPE, None], optional):
            Where to export the predicted map. Defaults to None.
        vis (bool, optional):
            Show the mesh model and predicted results. Defaults to False.
    """

    if isinstance(IDs_to_labels, str):
        IDs_to_labels = {
            int(k): v for k, v in json.load(open(IDs_to_labels, "r")).items()
        }

    ## Create the camera set
    # Do the camera operations first because they are fast and good initial error checking
    camera_set = MetashapeCameraSet(
        cameras_file,
        image_folder,
        original_image_folder=original_image_folder,
        validate_images=True,
    )

    # If the ROI is not None, subset to cameras within a buffer distance of the ROI
    # TODO let get_subset_ROI accept a None ROI and return the full camera set
    if subset_images_folder is not None:
        camera_set = camera_set.get_cameras_in_folder(subset_images_folder)

    # Subset based on regex if requested
    if filename_regex is not None:
        camera_set = camera_set.get_cameras_matching_filename_regex(
            filename_regex=filename_regex
        )

    # If you only want to take every nth camera, helpful for initial testing
    if take_every_nth_camera is not None:
        camera_set = camera_set.get_subset_cameras(
            range(0, len(camera_set), take_every_nth_camera)
        )

    if ROI is not None and ROI_buffer_radius_meters is not None:
        # Extract cameras near the training data
        camera_set = camera_set.get_subset_ROI(
            ROI=ROI, buffer_radius=ROI_buffer_radius_meters
        )

    # If the number of aggregation clusters is not set but the number of cameras per cluster is,
    # then compute it
    if n_aggregation_clusters is None and n_cameras_per_aggregation_cluster is not None:
        n_aggregation_clusters = int(
            math.ceil(len(camera_set) / n_cameras_per_aggregation_cluster)
        )

    # Choose whether to use a mesh class that aggregates by clusters of cameras and chunks of the mesh
    MeshClass = (
        TexturedPhotogrammetryMesh
        if n_aggregation_clusters is None
        else TexturedPhotogrammetryMeshChunked
    )
    ## Create the mesh
    mesh = MeshClass(
        mesh_file,
        input_CRS=mesh_CRS,
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

    # Create the potentially-empty dict of kwargs to match what this class expects
    n_clusters_kwargs = (
        {} if n_aggregation_clusters is None else {"n_clusters": n_aggregation_clusters}
    )

    ## Perform aggregation, this is the slow step
    aggregated_face_labels, _ = mesh.aggregate_projected_images(
        segmentor_camera_set,
        aggregate_img_scale=aggregate_image_scale,
        **n_clusters_kwargs,
    )

    # If requested, save this data
    if aggregated_face_values_savefile is not None:
        ensure_containing_folder(aggregated_face_values_savefile)
        np.save(aggregated_face_values_savefile, aggregated_face_labels)

    # Find the index of the most common class per face, with faces with no predictions set to nan
    predicted_face_classes = find_argmax_nonzero_value(
        aggregated_face_labels, keepdims=True
    )

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
        ensure_containing_folder(predicted_face_classes_savefile)
        np.save(predicted_face_classes_savefile, predicted_face_classes)

    if vis:
        # Show the mesh with predicted classes
        mesh.vis(vis_scalars=predicted_face_classes)

    # If the vector file should be exported
    if top_down_vector_projection_savefile is not None:
        # Compute the label names
        if IDs_to_labels is not None:
            # This ensures that any missing keys are replaced with None so proper indexing is retained
            label_names = [
                IDs_to_labels.get(i, None)
                for i in range(max(list(IDs_to_labels.keys())) + 1)
            ]
        else:
            label_names = None
        # Export the 2D top down projection
        mesh.export_face_labels_vector(
            face_labels=np.squeeze(predicted_face_classes),
            export_file=top_down_vector_projection_savefile,
            vis=vis,
            label_names=label_names,
        )


def parse_args():
    description = (
        "This script aggregates predictions from individual images onto the mesh. This aggregated "
        + "prediction can then be exported into geospatial coordinates. The default option is to "
        + "use the provided example data. All of the arguments are passed to "
        + "geograypher.entrypoints.workflow_functions.aggregate_images "
        + "which has the following documentation:\n\n"
        + aggregate_images.__doc__
    )
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description=description
    )
    parser.add_argument("--mesh-file", required=True)
    parser.add_argument("--cameras-file", required=True)
    parser.add_argument("--image-folder", required=True)
    parser.add_argument("--label-folder", required=True)
    parser.add_argument("--mesh-CRS", required=True)
    parser.add_argument("--original-image-folder", type=Path)
    parser.add_argument("--subset-images-folder", type=Path)
    parser.add_argument("--take-every-nth-camera", type=int)
    parser.add_argument("--DTM-file", type=Path)
    parser.add_argument("--height-above-ground-threshold", type=float, default=2)
    parser.add_argument("--ROI")
    parser.add_argument("--ROI-buffer-radius-meters", default=50, type=float)
    parser.add_argument("--IDs-to-labels", default=EXAMPLE_IDS_TO_LABELS)
    parser.add_argument("--mesh-downsample", type=float, default=1.0)
    parser.add_argument("--aggregate-image-scale", type=float, default=0.25)
    parser.add_argument("--n-aggregation-clusters", type=int)
    parser.add_argument("--aggregated-face-values-savefile", type=Path)
    parser.add_argument("--predicted-face-classes-savefile", type=Path)
    parser.add_argument(
        "--top-down-vector-projection-savefile", default="vis/predicted_map.geojson"
    )
    parser.add_argument("--vis", action="store_true")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # Parse command line args
    args = parse_args()
    # Pass command line args to aggregate_images
    aggregate_images(**args.__dict__)
