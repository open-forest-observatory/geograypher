import typing
from pathlib import Path

import geopandas as gpd
import numpy as np

from geograypher.constants import PATH_TYPE, PRED_CLASS_ID_KEY
from geograypher.meshes import TexturedPhotogrammetryMesh
from geograypher.utils.files import ensure_containing_folder


def label_polygons(
    mesh_file: PATH_TYPE,
    mesh_transform_file: PATH_TYPE,
    aggregated_face_values_file: PATH_TYPE,
    geospatial_polygons_to_label: typing.Union[PATH_TYPE, None],
    geospatial_polygons_labeled_savefile: typing.Union[PATH_TYPE, None],
    mesh_downsample: float = 1.0,
    DTM_file: typing.Union[PATH_TYPE, None] = None,
    height_above_ground_threshold: float = 2.0,
    ground_voting_weight: float = 0.01,
    ROI: typing.Union[PATH_TYPE, None] = None,
    ROI_buffer_radius_meters: float = 50,
    IDs_to_labels: typing.Union[dict, None] = None,
):
    """
    Label each polygon with the most commonly predicted class as computed by the weighted sum of 3D
    face areas

    Args:
        mesh_file (PATH_TYPE):
            Path to the Metashape-exported mesh file
        mesh_transform_file (PATH_TYPE):
            Transform from the mesh coordinates to the earth-centered, earth-fixed frame. Can be a
            4x4 matrix represented as a .csv, or a Metashape cameras file containing the information.
        aggregated_face_values_file (PATH_TYPE):
            Path to a (n_faces, n_classes) numpy array containing the frequency of each class
            prediction for each face
        geospatial_polygons_to_label (typing.Union[PATH_TYPE, None], optional):
            Each polygon/multipolygon will be labeled independently. Defaults to None.
        geospatial_polygons_labeled_savefile (typing.Union[PATH_TYPE, None], optional):
            Where to save the labeled results.
        mesh_downsample (float, optional):
            Fraction to downsample mesh. Should match what was used to generate the
            aggregated_face_values_file. Defaults to 1.0.
        DTM_file (typing.Union[PATH_TYPE, None], optional):
            Path to a digital terrain model file to remove ground points. Defaults to None.
        height_above_ground_threshold (float, optional):
            Height in meters above the DTM to consider ground. Only used if DTM_file is set.
            Defaults to 2.0.
        ground_voting_weight (float, optional):
            Faces identified as ground are given this weight during voting. Defaults to 0.01.
        ROI (typing.Union[PATH_TYPE, None], optional):
            Geofile region of interest to crop the mesh to. Should match what was used to generate
            aggregated_face_values_file. Defaults to None.
        ROI_buffer_radius_meters (float, optional):
            Keep points within this distance of the provided ROI object, if unset, everything will
            be kept. Should match what was used to generate aggregated_face_values_file. Defaults to 50.
        IDs_to_labels (typing.Union[dict, None], optional):
            Mapping from integer IDs to human readable labels. Defaults to None.
    """
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

    # Extract which vertices are labeled as ground
    # TODO check that the types are correct here
    ground_mask_verts = mesh.get_height_above_ground(
        DTM_file=DTM_file,
        threshold=height_above_ground_threshold,
    )
    # Convert that vertex labels into face labels
    ground_mask_faces = mesh.vert_to_face_texture(ground_mask_verts)

    # Ground points get a weighting of ground_voting_weight, others get 1
    ground_weighting = 1 - (
        (1 - ground_voting_weight) * ground_mask_faces.astype(float)
    )
    # Perform per-polygon labeling
    polygon_labels = mesh.label_polygons(
        face_labels=predicted_face_classes,
        polygons=geospatial_polygons_to_label,
        face_weighting=ground_weighting,
    )

    # Save out the predicted classes into a copy of the original file
    geospatial_polygons = gpd.read_file(geospatial_polygons_to_label)
    geospatial_polygons[PRED_CLASS_ID_KEY] = polygon_labels
    ensure_containing_folder(geospatial_polygons_labeled_savefile)
    geospatial_polygons.to_file(geospatial_polygons_labeled_savefile)
