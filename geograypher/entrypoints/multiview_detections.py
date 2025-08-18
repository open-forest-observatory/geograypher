"""
TODO
"""

import argparse
import logging
from pathlib import Path
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas
import pyproj
from shapely.geometry import Point

from geograypher.cameras import MetashapeCameraSet
from geograypher.constants import LAT_LON_CRS
from geograypher.meshes.meshes import TexturedPhotogrammetryMesh
from geograypher.predictors.derived_segmentors import RegionDetectionSegmentor
from geograypher.utils.files import ensure_folder


TRANSFORMS = {
    None: None,
    "square": lambda x: x**2,
    "cube": lambda x: x**3,
}


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Triangulate tree locations from image segmentations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        required=True,
        help="Directory containing raw images TODO: Nested?",
    )
    parser.add_argument(
        "--gpkg-dir",
        type=Path,
        required=True,
        help="Directory containing .gpkg files (one per image) TODO: Nested?",
    )
    parser.add_argument(
        "--camera-file",
        type=Path,
        required=True,
        help="Path to camera XML file containing camera locations from photogrammetry.",
    )
    parser.add_argument(
        "--mesh-file",
        type=Path,
        required=True,
        help="Path to georeferenced mesh file.",
    )
    parser.add_argument(
        "--mesh-crs",
        type=int,
        required=True,
        help="The CRS to interpret the mesh in (integer).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for triangulated tree locations.",
    )
    parser.add_argument(
        "--original-image-folder",
        type=Path,
        help="If provided, this will be subtracted off the beginning of absolute image paths"
        " stored in the camera_file. See MetashapeCameraSet for details.",
    )
    parser.add_argument(
        "--image-extension", type=str, default=".JPG", help="Image file extension"
    )
    parser.add_argument(
        "--similarity-threshold-meters",
        type=float,
        default=4.0,
        help="Ray intersection threshold in meters",
    )
    parser.add_argument(
        "--louvain-resolution",
        type=float,
        default=2.0,
        help="Louvain resolution parameter, larger value = smaller communities",
    )
    parser.add_argument(
        "--nonlinearity",
        choices=TRANSFORMS.keys(),
        default=None,
        help="Nonlinear transform to apply to distances before inverting and"
        " creating the graph weights. The default graph weight is 1/x, where x"
        " is the intersection distance, but with --nonlinearity square the"
        " graph weight would instead be 1/x**2.",
    )
    args = parser.parse_args()

    # Check inputs
    assert args.images_dir.is_dir(), f"Images dir {args.images_dir} doesn't exist"
    assert args.gpkg_dir.is_dir(), f"Gpkg dir {args.gpkg_dir} doesn't exist"
    assert args.camera_file.is_file(), f"Camera XML {args.camera_file} doesn't exist"
    assert args.mesh_file.is_file(), f"Mesh {args.mesh_file} doesn't exist"

    # Ensure output directory exists
    ensure_folder(args.output_dir)

    return args


def multiview_detections(
    images_dir: Path,
    gpkg_dir: Path,
    camera_file: Path,
    mesh_file: Path,
    mesh_crs: pyproj.crs.CRS,
    output_dir: Path,
    original_image_folder: Optional[Path],
    image_file_extension: str = ".JPG",
    similarity_threshold_meters: float = 0.1,
    louvain_resolution: float = 2.0,
    transform=None,
):
    """
    TODO
    """

    logger = logging.getLogger(__name__)

    # Load camera set
    camera_set = MetashapeCameraSet(
        camera_file=camera_file,
        image_folder=images_dir,
        original_image_folder=original_image_folder,
        validate_images=True,
    )

    # Load mesh in ECEF (EPSG:4978). This is because the camera locations are defined
    # in the photogrammetry reference frame (PRF), and we can convert easily between
    # ECEF and the PRF
    mesh = TexturedPhotogrammetryMesh(mesh_file, input_CRS=mesh_crs)
    mesh.reproject_CRS(target_CRS=pyproj.crs.CRS.from_epsg(4978), inplace=True)

    # TODO: Add detail about covering
    ceiling, floor = mesh.export_covering_meshes(N=80, z_buffer=(0, 1), subsample=2)

    # Convert to local photogrammetry frame and save
    epsg_4978_to_local = np.linalg.inv(camera_set.get_local_to_epsg_4978_transform())
    for bmesh, name in [
        (ceiling, "boundary_ceiling.ply"),
        (floor, "boundary_floor.ply"),
    ]:
        bmesh.transform(epsg_4978_to_local, inplace=True)
        bmesh.save(output_dir / name)
    logger.info("Boundary meshes saved")

    # Load region detector
    logger.info("Loading region detection segmentor")
    detector = RegionDetectionSegmentor(
        base_folder=images_dir,
        lookup_folder=gpkg_dir,
        label_key=None,
        class_map=None,
    )

    # Triangulate detections (tree locations) and add lines/points to plotter
    tree_points = camera_set.triangulate_detections(
        detector=detector,
        transform_to_epsg_4978=camera_set.get_local_to_epsg_4978_transform(),
        boundaries=(ceiling, floor),
        limit_ray_length_meters=160,
        limit_angle_from_vert=np.deg2rad(50),
        similarity_threshold_meters=similarity_threshold_meters,
        transform=transform,
        louvain_resolution=louvain_resolution,
        out_dir=output_dir,
    )

    # Save results
    gpkg_path = output_dir / "tree_locations.gpkg"
    # Note that we want to store (lat, lon) - which is what tree_points
    # comes in as - in the form (y, x)
    gdf = gpd.GeoDataFrame(
        pandas.DataFrame(tree_points, columns=["y", "x", "z"]),
        geometry=[Point(x, y, z) for y, x, z in tree_points],
        crs=LAT_LON_CRS,
    )
    gdf.to_file(gpkg_path)
    logger.info(f"Saved triangulated tree locations to {gpkg_path}")


if __name__ == "__main__":
    args = parse_args()
    multiview_detections(
        images_dir=args.images_dir,
        gpkg_dir=args.gpkg_dir,
        camera_file=args.camera_file,
        mesh_file=args.mesh_file,
        mesh_crs=pyproj.crs.CRS.from_epsg(args.mesh_crs),
        output_dir=args.output_dir,
        original_image_folder=args.original_image_folder,
        image_file_extension=args.image_extension,
        similarity_threshold_meters=args.similarity_threshold_meters,
        louvain_resolution=args.louvain_resolution,
        transform=TRANSFORMS[args.nonlinearity],
    )
