import argparse
import cProfile
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyvista as pv
from shapely.geometry import Point

from geograypher.cameras import MetashapeCameraSet
from geograypher.meshes.meshes import TexturedPhotogrammetryMesh
from geograypher.predictors.derived_segmentors import RegionDetectionSegmentor
from geograypher.utils.files import ensure_folder


def main(
    images_dir: Path,
    gpkg_dir: Path,
    camera_xml: Path,
    mesh_file: Path,
    output_dir: Path,
    image_file_extension: str = ".JPG",
    similarity_threshold_meters: float = 0.1,
    louvain_resolution: float = 2.0,
):
    # Load camera set
    cameras = MetashapeCameraSet(camera_file=camera_xml, image_folder=images_dir)

    # THIS IS A HACK WHILE WE ARE RUNNING WITH IMAGE SUBSETS
    imset = set([path.name for path in images_dir.glob(f"*{image_file_extension}")])
    subset_cameras = [
        cam for cam in cameras.cameras if Path(cam.image_filename).name in imset
    ]
    cameras.cameras = subset_cameras

    # Load mesh
    mesh = TexturedPhotogrammetryMesh(
        mesh_file, transform_filename=camera_xml, require_transform=True
    )
    ceiling, floor = mesh.export_covering_meshes(N=20, z_buffer_m=(2, 0), subsample=2)
    ceiling.save(output_dir / "b2_ceiling.ply")
    floor.save(output_dir / "b2_floor.ply")
    print("Boundary meshes saved")

    # Load region detector
    print("Loading region detection segmentor")
    detector = None  # Use when line segments are precalculated
    detector = RegionDetectionSegmentor(
        detection_file_or_folder=gpkg_dir,
        image_file_extension=image_file_extension,
        use_absolute_filepaths=True,
        image_folder=images_dir,
    )

    # Triangulate detections (tree locations) and add lines/points to plotter
    tree_points = cameras.triangulate_detections(
        detector=detector,
        boundaries=(ceiling, floor),
        transform_to_epsg_4978=mesh.local_to_epgs_4978_transform,
        similarity_threshold_meters=similarity_threshold_meters,
        louvain_resolution=louvain_resolution,
        out_dir=output_dir,
        # line_segments_file=output_dir / "line_segments.npz",
        # positive_edges_file=output_dir / "positive_edges.json",
        # communities_file=output_dir / "communities.npz",
        # vis_dir=output_dir,
    )

    # Save results as CSV and GeoJSON
    csv_path = output_dir / "tree_locations.csv"
    geojson_path = output_dir / "tree_locations.geojson"
    df = pd.DataFrame(tree_points, columns=["x", "y", "z"])
    df.to_csv(csv_path, index=False)
    gdf = gpd.GeoDataFrame(
        df, geometry=[Point(x, y, z) for x, y, z in tree_points], crs="EPSG:4978"
    )
    gdf.to_file(geojson_path, driver="GeoJSON")

    print(f"Saved triangulated tree locations to {csv_path} and {geojson_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Triangulate tree locations from image segmentations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("images_dir", type=Path, help="Directory containing raw images")
    parser.add_argument(
        "gpkg_dir", type=Path, help="Directory containing .gpkg files (one per image)"
    )
    parser.add_argument("camera_xml", type=Path, help="Path to camera XML file")
    parser.add_argument("mesh_file", type=Path, help="Path to mesh file")
    parser.add_argument(
        "output_dir", type=Path, help="Output directory for triangulated tree locations"
    )
    parser.add_argument(
        "--image_file_extension", type=str, default=".JPG", help="Image file extension"
    )
    parser.add_argument(
        "--similarity_threshold_meters",
        type=float,
        default=4.0,
        help="Ray intersection threshold in meters",
    )
    parser.add_argument(
        "--louvain_resolution",
        type=float,
        default=2.0,
        help="Louvain community resolution parameter",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    assert (
        args.images_dir.is_dir()
    ), f"Images directory {args.images_dir} does not exist"
    assert args.gpkg_dir.is_dir(), f"Gpkg directory {args.gpkg_dir} does not exist"
    assert (
        args.camera_xml.is_file()
    ), f"Camera XML file {args.camera_xml} does not exist"
    assert args.mesh_file.is_file(), f"Mesh file {args.mesh_file} does not exist"

    # Ensure output directory exists
    ensure_folder(args.output_dir)

    profile = cProfile.Profile()
    profile.enable()
    main(
        images_dir=args.images_dir,
        gpkg_dir=args.gpkg_dir,
        camera_xml=args.camera_xml,
        mesh_file=args.mesh_file,
        output_dir=args.output_dir,
        image_file_extension=args.image_file_extension,
        similarity_threshold_meters=args.similarity_threshold_meters,
        louvain_resolution=args.louvain_resolution,
    )
    profile.disable()
    profile.dump_stats(args.output_dir / f"profile_{int(time.time() * 1e6)}.snakeviz")
