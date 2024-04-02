import typing
from pathlib import Path

import geopandas as gpd
import numpy as np

from geograypher.constants import PATH_TYPE, PRED_CLASS_ID_KEY
from geograypher.meshes import TexturedPhotogrammetryMesh


def label_polygons(
    mesh_file: PATH_TYPE,
    mesh_transform_file: PATH_TYPE,
    aggregated_face_values_file: PATH_TYPE,
    mesh_downsample: float = 1.0,
    DTM_file: typing.Union[PATH_TYPE, None] = None,
    height_above_ground_threshold: float = 2.0,
    ROI: typing.Union[PATH_TYPE, None] = None,
    ROI_buffer_radius_meters: float = 50,
    IDs_to_labels: typing.Union[dict, None] = None,
    geospatial_polygons_to_label: typing.Union[PATH_TYPE, None] = None,
    geospatial_polygons_labeled_savefile: typing.Union[PATH_TYPE, None] = None,
):
    # Load this first because it's quick
    aggregated_face_values = np.load(aggregated_face_values_file)
    predicted_face_classes = np.argmax(aggregated_face_values, axis=1)

    ## Create the mesh
    mesh = TexturedPhotogrammetryMesh(
        mesh_file,
        transform_filename=mesh_transform_file,
        ROI=ROI,
        ROI_buffer_meters=ROI_buffer_radius_meters,
        IDs_to_labels=IDs_to_labels,
        downsample_target=mesh_downsample,
    )

    # TODO check types here
    ground_mask_verts = mesh.get_height_above_ground(
        DTM_file=DTM_file,
        threshold=height_above_ground_threshold,
    )
    ground_mask_faces = mesh.vert_to_face_texture(ground_mask_verts)

    # Ground points get a weighting of 0.01, others get 1
    # TODO make this weight tunable
    ground_weighting = 1 - (1 - 0.01) * ground_mask_faces.astype(float)
    polygon_labels = mesh.label_polygons(
        face_labels=predicted_face_classes,
        polygons=geospatial_polygons_to_label,
        face_weighting=ground_weighting,
    )
    geospatial_polygons = gpd.read_file(geospatial_polygons_to_label)
    geospatial_polygons[PRED_CLASS_ID_KEY] = polygon_labels
    Path(geospatial_polygons_labeled_savefile).mkdir(parents=True, exist_ok=True)
    geospatial_polygons.to_file(geospatial_polygons_labeled_savefile)
