import os
from pathlib import Path

import numpy as np
from imageio import imread
from scipy.sparse import load_npz, save_npz

from geograypher.cameras import MetashapeCameraSet
from geograypher.cameras.segmentor import SegmentorPhotogrammetryCameraSet
from geograypher.constants import PATH_TYPE
from geograypher.meshes.derived_meshes import TexturedPhotogrammetryMeshIndexPredictions
from geograypher.predictors.derived_segmentors import TabularRectangleSegmentor
from geograypher.utils.files import ensure_containing_folder


def project_detections(
    mesh_filename: PATH_TYPE,
    cameras_filename: PATH_TYPE,
    project_to_mesh: bool = False,
    convert_to_geospatial: bool = False,
    image_folder: PATH_TYPE = None,
    detections_folder: PATH_TYPE = None,
    projections_to_mesh_filename: PATH_TYPE = None,
    projections_to_geospatial_savefilename: PATH_TYPE = None,
    default_focal_length: float = None,
    image_shape: tuple = None,
    segmentor_kwargs: dict = {},
    vis_mesh: bool = False,
    vis_geodata: bool = False,
):
    """Project per-image detections to geospatial coordinates

    Args:
        mesh_filename (PATH_TYPE):
            Path to mesh file, in local coordinates from Metashape
        cameras_filename (PATH_TYPE):
            Path to cameras file. This also contains local-to-global coordinate transform to convert
            the mesh to geospatial units.
        project_to_mesh (bool, optional):
            Execute the projection to mesh step. Defaults to False.
        convert_to_geospatial (bool, optional):
            Execute the conversion to geospatial step. Defaults to False.
        image_folder (PATH_TYPE, optional):
            Path to the folder of images used to generate the detections. TODO, see if this can be
            removed since none of this information is actually used. Defaults to None.
        detections_folder (PATH_TYPE, optional):
            Folder of detections in the DeepForest format, one per image. Defaults to None.
        projections_to_mesh_filename (PATH_TYPE, optional):
            Where to save and/or load from the data for the detections projected to the mesh faces.
            Defaults to None.
        projections_to_geospatial_savefilename (PATH_TYPE, optional):
            Where to export the geospatial detections. Defaults to None.
        default_focal_length (float, optional):
            Since the focal length is not provided in many cameras files, it can be specified.
            The units are in pixels. TODO, figure out where this information can be reliably obtained
            from. Defaults to None.
        segmentor_kwargs (dict, optional):
            Dict of keyword arguments to pass to the segmentor. Defaults to {}.
        vis_mesh (bool, optional):
            Show the mesh with detections projected onto it. Defaults to False.
        vis_geodata (bool, optional):
            Show the geospatial projection. Defaults to False.

    Raises:
        ValueError: If convert_to_geospatial but no projections to mesh are available
        FileNotFoundError: If the projections_to_mesh_filename is set and needed but not present
    """
    # Create the mesh object, which will be used for either workflow
    mesh = TexturedPhotogrammetryMeshIndexPredictions(
        mesh_filename, transform_filename=cameras_filename
    )

    # Project per-image detections to the mesh
    if project_to_mesh:
        # Create a camera set associated with the images that have detections
        camera_set = MetashapeCameraSet(
            cameras_filename,
            image_folder,
            default_sensor_params={"f": default_focal_length, "cx": 0, "cy": 0},
        )
        # Infer the image shape from the first image in the folder
        if image_shape is None:
            image_filename_list = list(Path(image_folder).glob("*.*"))
            if len(image_filename_list) > 0:
                image_shape = imread(image_filename_list[0]).shape[:2]
            else:
                raise ValueError(
                    f"No image_shape provided and folder of images {image_folder} was empty"
                )
        # Create an object that looks up the detections from a folder of CSVs or one individual one.
        # Using this, it can generate "predictions" for a given image.
        detections_predictor = TabularRectangleSegmentor(
            detection_file_or_folder=detections_folder,
            image_folder=image_folder,
            image_shape=image_shape,
            **segmentor_kwargs,
        )

        # Wrap the camera set so that it returns the detections rather than the original images
        detections_camera_set = SegmentorPhotogrammetryCameraSet(
            camera_set, segmentor=detections_predictor
        )
        # Project the detections to the mesh
        aggregated_projections, _ = mesh.aggregate_projected_images(
            cameras=detections_camera_set, n_classes=detections_predictor.num_classes
        )

        # If a file is provided, save the
        if projections_to_mesh_filename is not None:
            # Export the per-face texture to an npz file, since it's a sparse array
            ensure_containing_folder(projections_to_mesh_filename)
            save_npz(projections_to_mesh_filename, aggregated_projections)

        if vis_mesh:
            # Determine which detection is predicted for each face, if any. In cases where multiple
            # detections project to the same face, the one with the lower index will be reported
            detection_ID_per_face = np.argmax(aggregated_projections, axis=1).astype(
                float
            )
            # Mask out locations for which there are no predictions
            detection_ID_per_face[np.sum(aggregated_projections, axis=1) == 0] = np.nan
            # Show the mesh
            mesh.vis(vis_scalars=detection_ID_per_face)

    # Convert per-face projections to geospatial ones
    if convert_to_geospatial:
        # Determine if the mesh texture was computed in the last step or otherwise if it can be loaded
        if not project_to_mesh:
            if projections_to_mesh_filename is None:
                raise ValueError("No projections_to_mesh_savefilename provided")
            elif os.path.isfile(projections_to_mesh_filename):
                aggregated_projections = load_npz(projections_to_mesh_filename)
            else:
                raise FileNotFoundError(
                    f"projections_to_mesh_filename {projections_to_mesh_filename} not found"
                )
        # Convert the per-face labels to geospatial coordinates. Optionally vis and/or export
        mesh.export_face_labels_vector(
            face_labels=aggregated_projections,
            export_file=projections_to_geospatial_savefilename,
            vis=vis_geodata,
        )
