import argparse
from pathlib import Path
from typing import List, Optional
import geopandas as gpd
import numpy as np
from PIL import Image, ImageDraw
from skimage.io import imread
from rasterio.features import rasterize
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Ground filter script.")
    parser.add_argument(
        "mask_im_dir", type=Path, help="Directory with ground mask images (png files)"
    )
    parser.add_argument("gpkg_dir", type=Path, help="Directory with GPKG files")
    parser.add_argument(
        "raw_im_dir", type=Path, help="Directory with original images (jpg files)"
    )
    parser.add_argument("out_dir", type=Path, help="Output directory")
    parser.add_argument(
        "--eval",
        help="Whether to produce additional evaluation images",
        action="store_true",
    )
    args = parser.parse_args()

    # Check that input directories exist
    assert args.mask_im_dir.is_dir(), f"{args.mask_im_dir} not found"
    assert args.gpkg_dir.is_dir(), f"{args.gpkg_dir} not found"
    assert args.raw_im_dir.is_dir(), f"{args.raw_im_dir} not found"
    # Create output directory if it doesn't exist
    if not args.out_dir.is_dir():
        args.out_dir.mkdir(parents=True)

    # Get all gpkg files and make sure the
    gpkg_files = list(args.gpkg_dir.glob("*.gpkg"))
    assert len(gpkg_files) > 0, f"No .gpkg files found in {args.gpkg_dir}"

    # For each gpkg file, check for matching png in im_dir
    for im_dir, extension in (args.mask_im_dir, "png"), (args.raw_im_dir, "JPG"):
        missing_images = []
        for gpkg in gpkg_files:
            image = im_dir / f"{gpkg.stem}.{extension}"
            if not image.is_file():
                missing_images.append(image)
        if missing_images:
            print("Missing image files for the following gpkg files:")
            for missing in missing_images:
                print(f"\t{missing}")
            assert (
                not missing_images
            ), "Some GPKG files do not have matching image files."

    return args


def detection_filter(
    gpkg: Path,
    png: Path,
    cutoff_fraction: float = 0.4,
) -> tuple[gpd.GeoDataFrame, List]:
    """
    Filters rows in a GPKG file based on the validity of the corresponding polygonal
    mask in an image.

    Args:
        gpkg (Path): Path to the input GPKG file containing polygon geometries.
        png (Path): Path to the PNG image file (ground mask) where 0=unknown, 1=ground,
            2=above ground.
        cutoff_fraction (float, optional): If the fraction of pixels in the mask that
            are valid is above this, keep that row

    Raises:
        AssertionError: If any geometry in the GPKG is not a Polygon.

    Returns:
        tuple[gpd.GeoDataFrame, List]:
            Filtered GeoDataFrame and the list of valid row indices
    """

    gdf = gpd.read_file(gpkg)

    # Load the mask images (0=unknown, 1=ground, 2=above ground)
    img = imread(str(png))
    # Use first channel if RGB
    if img.ndim == 3:
        img = img[..., 0]

    # Raster each gpkg row (a detection polygon) onto a mask, and check that
    # against the image. detections that are mostly valid will be kept and returned
    good_row_indices = []
    for idx, row in gdf.iterrows():
        geom = row.geometry
        assert geom.geom_type in {"Polygon", "MultiPolygon"}, \
            f"Process won't work with non-polygons ({geom.geom_type})"

        # If it's a MultiPolygon, split into individual polygons
        if geom.geom_type == "MultiPolygon":
            shapes = [(poly, 1) for poly in geom.geoms]
        else:
            shapes = [(geom, 1)]

        mask = rasterize(
            shapes,
            out_shape=img.shape,
            fill=0,
            dtype=np.uint8,
        ).astype(bool)

        # Grab pixel vector of shape (N,). Note that the image is single channel, so
        # this is not (N, 3) as an RBG image would be
        pixels = img[mask]
        if pixels.size == 0:
            continue
        valid = pixels == 2
        if np.sum(valid) / pixels.size > cutoff_fraction:
            good_row_indices.append(idx)

    return gdf.loc[good_row_indices], good_row_indices


def eval_image(
    gpkg: Path, raw: Path, mask: Path, good_row_indices: List, save_path: Path
) -> None:

    gdf = gpd.read_file(gpkg)
    vis_img = imread(str(raw))
    pil_img = Image.fromarray(vis_img.astype(np.uint8), mode="RGB").convert("RGBA")

    mask_img = imread(str(mask))
    # Use first channel if RGB
    if mask_img.ndim == 3:
        mask_img = mask_img[..., 0]

    overlay = np.zeros((pil_img.height, pil_img.width, 4), dtype=np.uint8)
    overlay[mask_img == 0] = (255, 0, 0, 50)
    overlay[mask_img == 1] = (0, 0, 255, 100)
    mask_overlay = Image.alpha_composite(pil_img, Image.fromarray(overlay, mode="RGBA"))

    # Draw polygons: green for kept, red for discarded
    overlay_draw = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay_draw, "RGBA")
    for idx, row in gdf.iterrows():
        color = (0, 255, 0, 70) if idx in good_row_indices else (255, 0, 0, 70)
        geom = row.geometry
        if geom.geom_type == "Polygon":
            polygons = [geom]
        elif geom.geom_type == "MultiPolygon":
            polygons = list(geom.geoms)
        else:
            raise NotImplementedError(f"Can't handle {type(geom)}")
        for poly in polygons:
            # Convert polygon to pixel coordinates
            coords = list(poly.exterior.coords)
            # coords are (x, y), but PIL expects (col, row) so we're good
            draw.polygon(coords, outline=(0, 0, 0, 255), fill=color)

    detection_overlay = Image.alpha_composite(pil_img, overlay_draw)

    for image, suffix in (mask_overlay, "_ground_mask"), (detection_overlay, "_detections"):
        path = save_path.parent / (save_path.stem + suffix + save_path.suffix)
        image.convert("RGB").save(path)


def main():

    args = parse_args()

    gpkg_files = list(args.gpkg_dir.glob("*.gpkg"))
    for gpkg in tqdm(gpkg_files, desc="Filtering GPKGs"):
        filtered_gdf, good_row_indices = detection_filter(
            gpkg,
            args.mask_im_dir / f"{gpkg.stem}.png",
        )
        if args.eval:
            eval_image(
                gpkg,
                raw=args.raw_im_dir / f"{gpkg.stem}.JPG",
                mask=args.mask_im_dir / f"{gpkg.stem}.png",
                good_row_indices=good_row_indices,
                save_path=args.out_dir / f"{gpkg.stem}.jpg",
            )
        filtered_gdf.to_file(args.out_dir / gpkg.name, driver="GPKG")


if __name__ == "__main__":
    main()
