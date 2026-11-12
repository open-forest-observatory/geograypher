import argparse
import json

import numpy as np
import pyproj
from matplotlib import pyplot as plt
import geopandas as gpd
from pathlib import Path

from geograypher.cameras.rig_cameras import create_rig_cameras_from_equirectangular
from geograypher.cameras.segmentor import SegmentorPhotogrammetryCameraSet
from geograypher.meshes import TexturedPhotogrammetryMesh
from geograypher.predictors.derived_segmentors import LookUpSegmentor
from geograypher.utils.indexing import find_argmax_nonzero_value
from geograypher.utils.visualization import show_segmentation_labels
from geograypher.utils.parsing import parse_metashape_mesh_metadata
from geograypher.utils.geospatial import convert_CRS_3D_points

from geograypher.constants import EARTH_CENTERED_EARTH_FIXED_CRS


def _json_arg(value):
    """Parse a CLI argument that is either a JSON string or a path to a JSON file."""
    if Path(value).is_file():
        with open(value) as f:
            return json.load(f)
    return json.loads(value)


def _ids_to_labels_arg(value):
    # JSON keys are always strings, so convert back to integer class IDs
    return {int(k): v for k, v in _json_arg(value).items()}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Project semantic segmentation predictions from equirectangular (360 degree) imagery onto a photogrammetry mesh."
    )
    parser.add_argument(
        "--aggregate-image-scale",
        type=float,
        default=0.25,
        help="Aggregate images at this scale resolution",
    )
    parser.add_argument(
        "--ids-to-labels",
        type=_ids_to_labels_arg,
        default={
            0: "BO",
            1: "CR",
            2: "SH",
            3: "HE",
            4: "GR",
            5: "LO",
            6: "MA",
            7: "SK",
        },
        help="Mapping from integer class IDs to label names, as a JSON string or path to a JSON file",
    )
    parser.add_argument(
        "--cameras-file",
        type=Path,
        default=Path(
            "/ofo-share/repos/david/under-canopy-mapping/data/photogrammetry_results/config-st-0077-gopro-360-photos-timelapse_cameras.xml"
        ),
        help="Path to the camera data from metashape",
    )
    parser.add_argument(
        "--mesh-file",
        type=Path,
        default=Path(
            "/ofo-share/repos/david/under-canopy-mapping/data/photogrammetry_results/config-st-0077-gopro-360-photos-timelapse_mesh.ply"
        ),
        help="Path to the mesh data from metashape",
    )
    parser.add_argument(
        "--mesh-metadata-file",
        type=Path,
        default=Path(
            "/ofo-share/repos/david/under-canopy-mapping/data/photogrammetry_results/metadata.xml"
        ),
        help="Path to the mesh metadata from metashape",
    )
    parser.add_argument(
        "--image-folder",
        type=Path,
        default=Path(
            "/ofo-share/repos/david/under-canopy-mapping/data/data_to_annotate/2024-ofo-gopro__st-0077-gopro-360-photos-timelapse-subset"
        ),
        help="Path to the images, resampled to frame camera representations",
    )
    parser.add_argument(
        "--predicted-image-labels-folder",
        type=Path,
        default=Path(
            "/ofo-share/repos/david/under-canopy-mapping/data/st_77_labels_2026_predictions"
        ),
        help="Path to the corresponding semantic segmentation predictions for the images",
    )
    parser.add_argument(
        "--original-image-folder",
        default="/ofo-share/repos/david/under-canopy-mapping/scratch/photogrammetry/st-0077-gopro-360-photos-timelapse",
        help="The location of the images when photogrammetry was run. Do not update this unless you use completely new photogrammetry data.",
    )
    parser.add_argument(
        "--vis-mesh-path",
        type=Path,
        default=Path("/tmp", "labeled_mesh.ply"),
        help="Output path for the labeled mesh",
    )
    parser.add_argument(
        "--vis-cameras-path",
        type=Path,
        default=Path("/tmp", "camera_rig_vis_mesh.ply"),
        help="Output path for the camera rig visualization mesh",
    )
    parser.add_argument(
        "--vis", action="store_true", help="Show intermediate visualizations"
    )
    parser.add_argument(
        "--rig-camera-def",
        type=_json_arg,
        default={
            "f": 1440 / 2,
            "cx": 0.0,
            "cy": 0.0,
            "image_width": 1440,
            "image_height": 1440,
        },
        help="Intrinsics of the perspective rig camera, as a JSON string or path to a JSON file",
    )
    parser.add_argument(
        "--rig-orientations",
        type=_json_arg,
        default=[
            {"yaw_deg": 0.0, "pitch_deg": 0.0, "roll_deg": 0.0},
            {"yaw_deg": 90.0, "pitch_deg": 0.0, "roll_deg": 0.0},
            {"yaw_deg": 180.0, "pitch_deg": 0.0, "roll_deg": 0.0},
            {"yaw_deg": 270.0, "pitch_deg": 0.0, "roll_deg": 0.0},
            {"yaw_deg": 0.0, "pitch_deg": -90.0, "roll_deg": 0.0},
            {"yaw_deg": 0.0, "pitch_deg": 90.0, "roll_deg": 0.0},
        ],
        help="List of rig camera orientations (yaw_deg, pitch_deg, roll_deg), as a JSON string or path to a JSON file",
    )
    parser.add_argument(
        "--resampled-format-str",
        default="_yaw{yaw_deg:03.0f}_pitch{pitch_deg:03.0f}",
        help="Format string for the filename suffix of the resampled perspective images",
    )
    return parser.parse_args()


def save_visualizations(
    mesh,
    camera_set,
    mesh_shift,
    mesh_CRS,
    predicted_face_classes,
    vis_mesh_path,
    vis_cameras_path,
):
    # Colormap the results and save them out
    # Get the mesh in the original CRS and shift
    pymesh = mesh.reproject_CRS(target_CRS=pyproj.CRS(mesh_CRS), inplace=False)
    pymesh.points -= np.array([mesh_shift[0], mesh_shift[1], 0])
    # Remove the previous coloring
    assert pymesh.n_cells == len(predicted_face_classes)
    del pymesh.point_data["RGB"]

    # Colormap the values
    colors = plt.colormaps["tab10"](predicted_face_classes.squeeze() / 10)
    # Rescale to 0-255 and convert to uint8
    colors = (colors * 255).astype(np.uint8)

    # Add the colors to the mesh and save
    pymesh.cell_data["RGB"] = colors
    pymesh.save(vis_mesh_path, texture="RGB")

    # # Save out the camera locations
    # Create a mesh representation of the camera locations for subsequent visualization
    # Get the camera mesh in the local coordinate system
    camera_mesh = camera_set.get_vis_mesh()
    # Get the transform from this frame to the earth-centered earth-fixed frame
    local_to_epsg_4978_transform = camera_set.get_local_to_epsg_4978_transform()
    # Apply the transform to get the mesh in ECEF
    camera_mesh.transform(local_to_epsg_4978_transform, inplace=True)

    # Project the points from ECEF to the CRS that the scene mesh was provided in
    reprojected_points = convert_CRS_3D_points(
        camera_mesh.points,
        input_CRS=EARTH_CENTERED_EARTH_FIXED_CRS,
        output_CRS=mesh_CRS,
    )
    camera_mesh.points = reprojected_points

    # Apply the same shift that was applied to the scene mesh to avoid quantization
    camera_mesh.points -= np.array([mesh_shift[0], mesh_shift[1], 0])
    # Save the camera mesh
    camera_mesh.save(vis_cameras_path)


def project_equirectangular_images(
    mesh_file,
    mesh_metadata_file,
    cameras_file,
    original_image_folder,
    predicted_image_labels_folder,
    rig_camera_def,
    rig_orientations,
    resampled_format_str,
    vis,
    ids_to_labels,
    image_folder=None,
    aggregate_image_scale=1.0,
    vis_mesh_path=None,
    vis_cameras_path=None,
):
    # The mesh file format does not support metadata so it is saved in a sidecar file. Since the
    # values of geospatial coordinates are large, quantization from the limits of float precision
    # have a noticable impact. To counteract this, the mesh is saved with zero-centered coordinates,
    # since float quantization is dependent on the scale of the value being represented. The metadata
    # file provides a shift to apply to the data to produce the original values. Also, it provides a
    # coordinate reference system (CRS) to interpret the values in.
    mesh_CRS, mesh_shift = parse_metashape_mesh_metadata(mesh_metadata_file)

    # Each original spherical camera is now represented by six perspective cameras all facing outward
    # from the center location.
    # Create camera set
    camera_set = create_rig_cameras_from_equirectangular(
        camera_file=cameras_file,
        original_images=original_image_folder,
        perspective_images=predicted_image_labels_folder,
        rig_camera=rig_camera_def,
        rig_orientations=rig_orientations,
        perspective_filename_format_str=resampled_format_str,
    )

    # And show
    if vis:
        camera_set.vis(frustum_scale=4.0)

    # Create a region of interest from the camera locations so we don't texture areas of the mesh
    # that are very far from cameras
    # Extract the region of interest around the camera locations
    ROI = camera_set.get_camera_locations(as_CRS=mesh_CRS)
    # Convert the ROI to a GeoDataFrame
    xs, ys = zip(*ROI)
    ROI = gpd.GeoDataFrame(geometry=gpd.points_from_xy(xs, ys), crs=mesh_CRS)

    # Show the region of interest
    if vis:
        ROI.plot()
        plt.show()

    # Load the mesh, clipped to 25 meters from the camera locations. Once it's loaded, show it.
    mesh = TexturedPhotogrammetryMesh(
        mesh_file,
        input_CRS=mesh_CRS,
        ids_to_labels=ids_to_labels,
        ROI=ROI,
        ROI_buffer_meters=25,
        shift=mesh_shift,
    )

    if vis:
        mesh.vis()

        # Show a set of semantic segmentation predictions to ensure they look correct. The predictions
        # are overlaid on the correponding images.

        show_segmentation_labels(
            label_folder=predicted_image_labels_folder,
            image_folder=image_folder,
            ids_to_labels=ids_to_labels,
            image_suffix=".png",
        )

    # This creates an object that behaves like a camera set but returns the segmented image instead
    # of the original RGB one.
    segmentor = LookUpSegmentor(
        base_folder=predicted_image_labels_folder,
        lookup_folder=predicted_image_labels_folder,
        num_classes=len(ids_to_labels),
    )

    # This wraps a camera set so it returns the corresponding segmentation label
    segmentor_camera_set = SegmentorPhotogrammetryCameraSet(
        camera_set, segmentor=segmentor
    )

    # Here the labels are finally projected to the mesh using techniques from computer graphics. Under the hood, this is performed with the OpenGL rendering stack.

    # You can skip the expensive aggregation step by loading a cached version
    # Otherwise, actually run agregation. This is slow, approximately 10-30 minutes depending on the system
    aggregated_face_labels, _ = mesh.aggregate_projected_images(
        segmentor_camera_set,
        aggregate_img_scale=aggregate_image_scale,
        apply_distortion=False,
    )

    if vis:
        # Compute the most commonly projected class per face and show it
        predicted_face_classes = find_argmax_nonzero_value(
            aggregated_face_labels, keepdims=True
        )
        mesh.vis(vis_scalars=predicted_face_classes, IDs_to_labels=ids_to_labels)


if __name__ == "__main__":
    args = parse_args()

    project_equirectangular_images(**args.__dict__)
