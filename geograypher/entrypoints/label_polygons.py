import argparse
import typing

import geopandas as gpd
import numpy as np
import pyproj

from geograypher.constants import PATH_TYPE, PRED_CLASS_ID_KEY
from geograypher.meshes.derived_meshes import TexturedPhotogrammetryMeshChunked
from geograypher.utils.files import ensure_containing_folder


def label_polygons(
    mesh_file: PATH_TYPE,
    input_CRS: pyproj.CRS,
    aggregated_face_values_file: PATH_TYPE,
    geospatial_polygons_to_label: typing.Union[PATH_TYPE, None],
    geospatial_polygons_labeled_savefile: typing.Union[PATH_TYPE, None],
    mesh_downsample: float = 1.0,
    DTM_file: typing.Union[PATH_TYPE, None] = None,
    height_above_ground_threshold: float = 2.0,
    ground_voting_weight: float = 0.01,
    ROI: typing.Union[PATH_TYPE, None] = None,
    ROI_buffer_radius_meters: float = 50,
    n_polygons_per_cluster: int = 1000,
    IDs_to_labels: typing.Union[dict, None] = None,
    vis_mesh: bool = False,
):
    """
    Label each polygon with the most commonly predicted class as computed by the weighted sum of 3D
    face areas

    Args:
        mesh_file (PATH_TYPE):
            Path to the Metashape-exported mesh file
        input_CRS (pyproj.CRS):
            The CRS to interpret the mesh in.
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
        n_polygons_per_cluster (int, optional):
            The number of polygons to use in each cluster, when computing labeling by chunks.
            Defaults to 1000.
        IDs_to_labels (typing.Union[dict, None], optional):
            Mapping from integer IDs to human readable labels. Defaults to None.
    """
    # Load this first because it's quick
    aggregated_face_values = np.load(aggregated_face_values_file)
    predicted_face_classes = np.argmax(aggregated_face_values, axis=1).astype(float)
    no_preds_mask = np.all(np.logical_not(np.isfinite(aggregated_face_values)), axis=1)
    predicted_face_classes[no_preds_mask] = np.nan

    ## Create the mesh
    mesh = TexturedPhotogrammetryMeshChunked(
        mesh_file,
        input_CRS=input_CRS,
        ROI=ROI,
        ROI_buffer_meters=ROI_buffer_radius_meters,
        IDs_to_labels=IDs_to_labels,
        downsample_target=mesh_downsample,
    )

    if vis_mesh:
        mesh.vis(vis_scalars=predicted_face_classes)

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
    if vis_mesh:
        ground_masked_predicted_face_classes = predicted_face_classes.copy()
        ground_masked_predicted_face_classes[ground_mask_faces.astype(bool)] = np.nan
        mesh.vis(vis_scalars=ground_masked_predicted_face_classes)

    # Perform per-polygon labeling
    polygon_labels = mesh.label_polygons(
        face_labels=predicted_face_classes,
        polygons=geospatial_polygons_to_label,
        face_weighting=ground_weighting,
        n_polygons_per_cluster=n_polygons_per_cluster,
    )

    # Save out the predicted classes into a copy of the original file
    geospatial_polygons = gpd.read_file(geospatial_polygons_to_label)
    geospatial_polygons[PRED_CLASS_ID_KEY] = polygon_labels
    ensure_containing_folder(geospatial_polygons_labeled_savefile)
    geospatial_polygons.to_file(geospatial_polygons_labeled_savefile)


def parse_args():
    description = (
        "This script labels indidual geospatial polgyons from per-face aggregated predictions. "
        + "All of the arguments are passed to "
        + "geograypher.entrypoints.label_polygons "
        + "which has the following documentation:\n\n"
        + label_polygons.__doc__
    )
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description=description
    )
    parser.add_argument("--mesh-file", required=True)
    parser.add_argument("--input-CRS", required=True)
    parser.add_argument("--aggregated-face-values-file")
    parser.add_argument("--geospatial-polygons-to-label")
    parser.add_argument("--geospatial-polygons-labeled-savefile")
    parser.add_argument("--mesh-downsample", type=float, default=1.0)
    parser.add_argument("--DTM-file")
    parser.add_argument("--height-above-ground-threshold", type=float, default=2)
    parser.add_argument("--ground-voting-weight", type=float, default=0.01)
    parser.add_argument("--ROI")
    parser.add_argument("--ROI-buffer-radius-meters", default=50, type=float)
    parser.add_argument("--IDs-to-labels", type=dict)
    parser.add_argument("--vis-mesh", action="store_true", default=False)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # Parse command line args
    args = parse_args()
    # Pass command line args to aggregate_images
    label_polygons(**args.__dict__)
