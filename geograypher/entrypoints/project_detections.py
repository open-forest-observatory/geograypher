import os

import numpy as np
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
    image_folder: PATH_TYPE = None,
    detections_folder: PATH_TYPE = None,
    projections_to_mesh_filename: PATH_TYPE = None,
    projections_to_geospatial_savefilename: PATH_TYPE = None,
    default_focal_length: float = None,
    project_to_mesh: bool = False,
    convert_to_geospatial: bool = False,
    vis_mesh: bool = False,
    vis_geodata: bool = False,
):
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
        )[:5]
        # Create an object that looks up the detections from a folder of CSVs. Using this, it can
        # generate "predictions" for a given image.
        detections_predictor = TabularRectangleSegmentor(
            pred_folder=detections_folder,
            image_folder=image_folder,
            label_key="instance_ID",
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
