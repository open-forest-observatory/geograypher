"""
Multiview tree detection and triangulation from aerial imagery.
"""

import argparse
import logging
from pathlib import Path
from typing import Optional

import geopandas as gpd
import matplotlib.cm as cm
import numpy as np
import pandas
import pyproj
import pyvista as pv
from matplotlib.colors import Normalize
from shapely.geometry import Point
from tqdm import tqdm

from geograypher.cameras import MetashapeCameraSet
from geograypher.constants import LAT_LON_CRS
from geograypher.meshes.meshes import TexturedPhotogrammetryMesh
from geograypher.predictors.derived_segmentors import RegionDetectionSegmentor
from geograypher.utils.files import ensure_folder
from geograypher.utils.geometric import get_scale_from_transform
from geograypher.utils.visualization import merge_cylinders

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
        help="Directory containing raw images or nested images",
    )
    parser.add_argument(
        "--detections-dir",
        type=Path,
        required=True,
        help="Directory containing .gpkg files (one per image), should"
        " match the nested structure of --images-dir",
    )
    parser.add_argument(
        "--camera-file",
        type=Path,
        required=True,
        help="Path to XML file containing camera calibrations and positions"
        " from photogrammetry software",
    )
    parser.add_argument(
        "--mesh-file",
        type=Path,
        required=True,
        help="Path to georeferenced mesh file",
    )
    parser.add_argument(
        "--mesh-crs",
        type=pyproj.crs.CRS.from_epsg,
        required=True,
        help="The CRS to interpret the mesh in (integer)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for triangulated tree locations",
    )
    parser.add_argument(
        "--original-image-folder",
        type=Path,
        help="If provided, this will be subtracted off the beginning of absolute image paths"
        " stored in the camera_file. See MetashapeCameraSet for details",
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
    parser.add_argument(
        "--vis",
        action="store_true",
        help="Generate 3D visualizations for detected trees and rays in --output-dir",
    )
    args = parser.parse_args()

    # Check inputs
    assert args.images_dir.is_dir(), f"Images dir {args.images_dir} doesn't exist"
    assert args.detections_dir.is_dir(), f"Gpkg dir {args.detections_dir} doesn't exist"
    assert args.camera_file.is_file(), f"Camera XML {args.camera_file} doesn't exist"
    assert args.mesh_file.is_file(), f"Mesh {args.mesh_file} doesn't exist"

    # Ensure output directory exists
    ensure_folder(args.output_dir)

    return args


def vis_lines(logger, line_segments_file, communities_file, out_dir, batch=250):

    data = np.load(line_segments_file)
    ray_starts = data["ray_starts"]
    ray_ends = data["ray_ends"]

    data = np.load(communities_file)
    community_IDs = data["ray_IDs"]
    community_points = data["community_points"]

    norm = Normalize(vmin=np.nanmin(community_IDs), vmax=np.nanmax(community_IDs))
    cmap = cm.get_cmap("tab20")
    cylinder_polydata = None
    n_batches = int(np.ceil(len(community_IDs) / batch))
    for i in tqdm(range(n_batches), desc="Building cylinders"):
        islice = slice(i * batch, min((i + 1) * batch, len(community_IDs)))
        batched = merge_cylinders(
            starts=ray_starts[islice],
            ends=ray_ends[islice],
            community_IDs=community_IDs[islice],
            cmap=cmap,
            norm=norm,
        )
        if cylinder_polydata is None:
            cylinder_polydata = batched
        else:
            cylinder_polydata = cylinder_polydata.merge(batched)

    if cylinder_polydata is not None:
        path = out_dir / "rays.ply"
        logger.info(f"Saving visualized cylinders to {path}")
        cylinder_polydata.save(path, texture="RGB")

    cube_polydata = None
    for comm_id, pt in enumerate(tqdm(community_points, desc="Building points")):
        cube = pv.Cube(center=pt, x_length=0.2, y_length=0.2, z_length=0.2)
        color = (np.array(cmap(norm(comm_id)))[:3] * 255).astype(np.uint8)
        cube.point_data["RGB"] = np.tile(color, (cube.n_points, 1))
        if cube_polydata is None:
            cube_polydata = cube
        else:
            cube_polydata = cube_polydata.merge(cube)

    if cube_polydata is not None:
        path = out_dir / "points.ply"
        logger.info(f"Saving visualized cubes to {path}")
        cube_polydata.save(path, texture="RGB")


def multiview_detections(
    images_dir: Path,
    detections_dir: Path,
    camera_file: Path,
    mesh_file: Path,
    mesh_crs: pyproj.crs.CRS,
    output_dir: Path,
    original_image_folder: Optional[Path],
    image_file_extension: str = ".JPG",
    similarity_threshold_meters: float = 0.1,
    louvain_resolution: float = 2.0,
    transform=None,
    vis: bool = False,
):
    """
    Triangulate tree locations from multi-view image segmentations.

    Args:
        images_dir (Path): Directory containing the raw images or nested images
        detections_dir (Path): Directory containing GeoPackage (.gpkg) files with object detection
            results, one file per corresponding image
        camera_file (Path): Path to XML file containing camera calibrations and positions
            from photogrammetry software
        mesh_file (Path): Path to the georeferenced 3D mesh file
        mesh_crs (pyproj.crs.CRS): Coordinate reference system for interpreting the mesh
        output_dir (Path): Directory where results will be saved
        original_image_folder (Optional[Path]): If provided, this will be subtracted off
            the beginning of absolute image paths stored in the --camera-file.
            See MetashapeCameraSet for details
        image_file_extension (str, optional): File extension for images. Defaults to ".JPG"
        similarity_threshold_meters (float, optional): Distance threshold in meters for
            considering ray intersections as similar. Defaults to 0.1
        louvain_resolution (float, optional): Resolution parameter for Louvain community
            detection clustering. Larger values create smaller clusters. Defaults to 2.0
        transform (callable, optional): Nonlinear transformation function to apply to
            intersection distances before creating graph weights. Defaults to None
        vis (bool, optional): Whether to generate 3D visualization files
            for detected trees and ray intersections. Defaults to False

    Returns:
        None: Results are saved to files in the output_dir:
            - tree_locations.gpkg: GeoDataFrame with triangulated tree positions in lat/lon
            - boundary_ceiling.ply and boundary_floor.ply: 3D mesh boundaries
            - rays.ply and points.ply: 3D visualizations (if vis=True)

    Raises:
        AssertionError: If input directories or files don't exist
    """

    logger = logging.getLogger(__name__)

    # Load camera set
    camera_set = MetashapeCameraSet(
        camera_file=camera_file,
        image_folder=images_dir,
        original_image_folder=original_image_folder,
        validate_images=True,
    )
    local_to_epsg_4978 = camera_set.get_local_to_epsg_4978_transform()

    # Load mesh in the photogrammetry reference frame (PRF). This is because the
    # camera locations are defined in the PRF
    mesh = TexturedPhotogrammetryMesh(mesh_file, input_CRS=mesh_crs)
    # Convert the mesh into one in the same coordinate frame as the camera set
    mesh.get_mesh_in_cameras_coords(camera_set, inplace=True)

    # Create boundary layers between the ground and the treetops that we
    # will check for ray intersections between. Note that the z_buffer
    # defines, in meters (which is then translated to local scale), the
    # spacing of the covering meshes from the mesh model. For example,
    # z_buffer[0] is the spacing between mesh and floor, and z_buffer[1]
    # is the spacing between mesh and ceiling.
    local_scale = 1 / get_scale_from_transform(local_to_epsg_4978)
    ceiling, floor = mesh.export_covering_meshes(
        N=50, z_buffer=(0, 1 * local_scale), subsample=2
    )
    mesh.save_mesh(output_dir / "mesh-local.ply")
    ceiling.save(output_dir / "boundary_ceiling.ply")
    floor.save(output_dir / "boundary_floor.ply")
    logger.info("Boundary meshes saved")

    # Load region detector
    logger.info("Loading region detection segmentor")
    detector = RegionDetectionSegmentor(
        base_folder=images_dir,
        lookup_folder=detections_dir,
        label_key=None,
        class_map=None,
    )

    # Triangulate detections (tree locations) and add lines/points to plotter
    tree_points = camera_set.triangulate_detections(
        detector=detector,
        boundaries=(ceiling, floor),
        limit_ray_length_meters=160,
        limit_angle_from_vert=np.deg2rad(50),
        similarity_threshold_meters=similarity_threshold_meters,
        transform=transform,
        louvain_resolution=louvain_resolution,
        out_dir=output_dir,
    )

    # Visualize the intersections
    if vis:
        vis_lines(
            logger,
            line_segments_file=output_dir / "line_segments.npz",
            communities_file=output_dir / "communities.npz",
            out_dir=output_dir,
        )

    # Save results
    gpkg_path = output_dir / "tree_locations.gpkg"
    # Note that we want to store (lat, lon) - which is what tree_points
    # comes in as - in the form (y, x)
    gdf = gpd.GeoDataFrame(
        geometry=[Point(x, y, z) for y, x, z in tree_points],
        crs=LAT_LON_CRS,
    )
    gdf.to_file(gpkg_path)
    logger.info(f"Saved triangulated tree locations to {gpkg_path}")


if __name__ == "__main__":
    args = parse_args()
    multiview_detections(
        images_dir=args.images_dir,
        detections_dir=args.detections_dir,
        camera_file=args.camera_file,
        mesh_file=args.mesh_file,
        mesh_crs=args.mesh_crs,
        output_dir=args.output_dir,
        original_image_folder=args.original_image_folder,
        image_file_extension=args.image_extension,
        similarity_threshold_meters=args.similarity_threshold_meters,
        louvain_resolution=args.louvain_resolution,
        transform=TRANSFORMS[args.nonlinearity],
        vis=args.vis,
    )
