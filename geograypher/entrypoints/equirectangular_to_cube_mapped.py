import argparse
from math import round
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from imageio.v2 import imread, imwrite
from sklearn.cluster import KMeans

from geograypher.cameras import MetashapeCameraSet
from geograypher.utils.geospatial import ensure_projected_CRS
from geograypher.utils.image import perspective_from_equirectangular

# Camera orientations (fov, yaw, pitch) to sample from each equirectangular image
FYPS = [
    (90 - 0.001, 0, -90),
    (90 - 0.001, 0, 90),
    (90 - 0.001, 0, 0),
    (90 - 0.001, 90, 0),
    (90 - 0.001, 180, 0),
    (90 - 0.001, 270, 0),
]

# Sample this many more pixels than the final resolution before downsampling
OVERSAMPLE_FACTOR = 4
# Order of interpolation
WARP_ORDER = 1
# One quarter of the width of the GoPro MAX2 equirectangular image
OUTPUT_SIZE = (1920, 1920)


def select_indices_nearest_cluster_centers(points, num_clusters):
    """Cluster points with kmeans and return, for each cluster, the index of the
    point closest to that cluster's center."""
    kmeans = KMeans(n_clusters=num_clusters).fit(points)
    closest_indices = [
        np.argmin(np.linalg.norm(points - center, axis=1))
        for center in kmeans.cluster_centers_
    ]
    return closest_indices


def chip_dataset(
    dataset,
    output_dir,
    fyps,
    n_images_to_save,
    output_size,
    oversample_factor,
    warp_order,
    photogrammetry_cameras_path=None,
):
    if photogrammetry_cameras_path is None:
        # Select files to save sequentially if no camera file is provided to determine locations
        files = sorted(dataset.glob("*"))

        # Select all images if no number is provided
        if n_images_to_save is None:
            files_to_save = files
        else:
            files_to_save = files[:: min(n_images_to_save, len(files))]
    else:
        # Use the locations to select a subset of images nearest the kmeans centers

        # Parse the photogrammetry result file using the geograypher camera class.
        # Default fields which are not actually used.
        cameras = MetashapeCameraSet(
            camera_file=photogrammetry_cameras_path,
            image_folder="",
            default_sensor_params={"f": 1.0, "cx": 0.0, "cy": 0.0},
        )

        # Extract the coordinates of the image locations and convert them to an (n, 2) array in an
        # appropriate projected CRS
        camera_locations_lon_lat = cameras.get_lon_lat_coords()
        camera_locations_lon_lat_gdf = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy(*zip(*camera_locations_lon_lat)),
            crs=4326,
        )
        camera_locations_gdf_projected = ensure_projected_CRS(
            camera_locations_lon_lat_gdf
        )
        camera_locations_numpy = np.vstack(
            [
                camera_locations_gdf_projected.geometry.x,
                camera_locations_gdf_projected.geometry.y,
            ]
        ).T

        # Run KMeans clustering and identify the image indices nearest each cluster center
        closest_indices = select_indices_nearest_cluster_centers(
            camera_locations_numpy, n_images_to_save
        )

        # Get all image filenames
        image_filenames = cameras.get_image_filename(index=None)

        # Return the subset corresponding to kmeans centers
        files_to_save = [image_filenames[i] for i in closest_indices]

    dataset_output_dir = output_dir / dataset.parts[-1]
    dataset_output_dir.mkdir(parents=True, exist_ok=True)

    last_img = None
    for f in files_to_save:
        last_img = imread(f)
        for fov, yaw, pitch in fyps:
            resampled = perspective_from_equirectangular(
                last_img,
                fov_deg=fov,
                output_size=output_size,
                yaw_deg=yaw,
                pitch_deg=pitch,
                oversample_factor=oversample_factor,
                warp_order=warp_order,
                return_mask=False,
            )
            out_path = (
                dataset_output_dir
                / f"{f.stem}_fov{int(round(fov))}_yaw{int(round(yaw))}_pitch{int(round(pitch))}.png"
            )
            imwrite(out_path, resampled.astype(np.uint8))

    return last_img


def visualize_mask_sum(
    img, fyps, output_size, oversample_factor, warp_order, output_path
):
    imgs_and_masks = [
        perspective_from_equirectangular(
            img,
            fov_deg=fov,
            output_size=output_size,
            yaw_deg=yaw,
            pitch_deg=pitch,
            warp_order=warp_order,
            oversample_factor=oversample_factor,
            return_mask=True,
        )
        for fov, yaw, pitch in fyps
    ]
    _, masks = zip(*imgs_and_masks)

    masks_sum = np.sum(masks, axis=0)
    plt.imshow(masks_sum)
    plt.colorbar()
    plt.savefig(output_path)
    print(f"Mask sum saved to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Chip equirectangular images into perspective views."
    )
    parser.add_argument(
        "image_dir",
        type=Path,
        help="Folder of equirectangular images to process",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory under which per-dataset output folders are created",
    )
    parser.add_argument(
        "--n-images-to-save",
        type=int,
        help=f"Chip this many original images. If not provided, all images will be saved. (default: None)",
    )
    parser.add_argument(
        "--oversample-factor",
        type=int,
        default=OVERSAMPLE_FACTOR,
        help=f"Sample this many times more pixels before downsampling (default: {OVERSAMPLE_FACTOR})",
    )
    parser.add_argument(
        "--warp-order",
        type=int,
        default=WARP_ORDER,
        help=f"Interpolation order for warping (default: {WARP_ORDER})",
    )
    parser.add_argument(
        "--output-size",
        type=int,
        nargs=2,
        metavar=("HEIGHT", "WIDTH"),
        default=list(OUTPUT_SIZE),
        help=f"Output image size in pixels (default: {OUTPUT_SIZE[0]} {OUTPUT_SIZE[1]})",
    )
    parser.add_argument(
        "--visualize-mask-sum-path",
        type=Path,
        help="Path to save the visualization of the summed sampling masks",
    )
    parser.add_argument(
        "--photogrammetry-cameras-path",
        help="Path to photogrammetry-produced camera file to use to sub-sample images geometrically",
    )
    args = parser.parse_args()

    return args


def main():
    args = parse_args

    output_size = tuple(args.output_size)

    last_img = None
    last_img = chip_dataset(
        args.image_dir,
        args.output_dir,
        FYPS,
        args.n_images_to_save,
        output_size,
        args.oversample_factor,
        args.warp_order,
        args.photogrammetry_cameras_path,
    )

    if args.visualize_mask_sum_path:
        if last_img is None:
            print("No images were processed; skipping mask sum visualization.")
        else:
            visualize_mask_sum(
                last_img,
                FYPS,
                output_size,
                args.oversample_factor,
                args.warp_order,
                args.visualize_mask_sum_path,
            )


if __name__ == "__main__":
    main()
