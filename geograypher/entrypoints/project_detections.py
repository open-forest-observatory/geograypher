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
    pred_mesh_texture_filename: PATH_TYPE,
    image_folder: PATH_TYPE = None,
    detections_folder: PATH_TYPE = None,
    pred_geospatial_filename: PATH_TYPE = None,
    default_focal_length: float = None,
    project: bool = True,
    export: bool = True,
    vis_mesh: bool = True,
    vis_geodata: bool = True,
):
    mesh = TexturedPhotogrammetryMeshIndexPredictions(
        mesh_filename, transform_filename=cameras_filename
    )

    if project:
        camera_set = MetashapeCameraSet(
            cameras_filename,
            image_folder,
            default_sensor_params={"f": default_focal_length, "cx": 0, "cy": 0},
        )
        segmentor = TabularRectangleSegmentor(
            pred_folder=detections_folder,
            image_folder=image_folder,
            label_key="instance_ID",
        )

        segmentor_camera_set = SegmentorPhotogrammetryCameraSet(
            camera_set, segmentor=segmentor
        )

        summed_projections, _ = mesh.aggregate_projected_images(
            cameras=segmentor_camera_set, n_classes=segmentor.num_classes
        )
        ensure_containing_folder(pred_mesh_texture_filename)
        save_npz(pred_mesh_texture_filename, summed_projections)

    if export:
        if not project:
            summed_projections = load_npz(pred_mesh_texture_filename)

        if vis_mesh:
            max_class = np.argmax(summed_projections, axis=1).astype(float)
            max_class[np.sum(summed_projections, axis=1) == 0] = np.nan
            mesh.vis(vis_scalars=max_class)

        mesh.export_face_labels_vector(
            face_labels=summed_projections,
            export_file=pred_geospatial_filename,
            vis=vis_geodata,
        )
