import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional, Union

import geopandas as gpd
import numpy as np
import rasterio as rio
from imageio import imwrite
from rasterio.features import rasterize
from rasterio.plot import reshape_as_image
from rasterio.transform import AffineTransformer
from rasterio.windows import Window
from shapely import Polygon
from tqdm import tqdm

from geograypher.constants import NULL_TEXTURE_INT_VALUE, PATH_TYPE
from geograypher.utils.files import ensure_containing_folder, ensure_folder
from geograypher.utils.io import read_image_or_numpy
from geograypher.utils.numeric import create_ramped_weighting


def create_windows(dataset_h_w, window_size, window_stride):
    windows = []
    for col_off in range(0, dataset_h_w[1], window_stride):
        for row_off in range(0, dataset_h_w[0], window_stride):
            windows.append(Window(col_off, row_off, window_size, window_size))
    return windows


def get_str_from_window(window: Window, raster_file, suffix):
    wd = window.todict()
    if suffix[0] != ".":
        suffix = "." + suffix
    window_str = f"{Path(raster_file).stem}:{wd['col_off']}:{wd['row_off']}:{wd['width']}:{wd['height']}{suffix}"
    return window_str


def parse_windows_from_files(
    files: list[Path], sep: str = ":", return_in_extent_coords: bool = True
) -> tuple[list[Window], Window]:
    """Return the boxes and extent from a list of filenames

    Args:
        files (list[Path]): List of filenames
        sep (str): Seperator between elements
        return_in_extent_coords (bool): Return in the coordinate frame of the extent

    Returns:
        tuple[list[Window], Window]: List of windows for each file and extent
    """
    # Split the coords out, currently ignorign the filename as the first element
    coords = [file.stem.split(sep)[1:] for file in files]

    # Compute the extents as the min/max of the boxes
    coords_array = np.array(coords).astype(int)

    xmin = np.min(coords_array[:, 0])
    ymin = np.min(coords_array[:, 1])
    xmax = np.max(coords_array[:, 2] + coords_array[:, 0])
    ymax = np.max(coords_array[:, 3] + coords_array[:, 1])
    extent = Window(row_off=ymin, col_off=xmin, width=xmax - xmin, height=ymax - ymin)

    if return_in_extent_coords:
        # Subtract out x and y min so it's w.r.t. the extent coordinates
        coords_array[:, 0] = coords_array[:, 0] - xmin
        coords_array[:, 1] = coords_array[:, 1] - ymin

    # Create windows from coords
    windows = [
        Window(
            col_off=coord[0],
            row_off=coord[1],
            width=coord[2],
            height=coord[3],
        )
        for coord in coords_array.astype(int)
    ]

    return windows, extent


def pad_to_full_size(img, desired_size):
    padding_size = np.array(desired_size) - np.array(img.shape[: len(desired_size)])
    if np.sum(padding_size) > 0:
        # Pad the trailing edge of the array only
        padding = [(0, width) for width in padding_size] + [(0, 0)] * (
            len(img.shape) - len(desired_size)
        )
        img = np.pad(img, padding)

    return img


def write_chips(
    raster_file: PATH_TYPE,
    output_folder: PATH_TYPE,
    chip_size: int,
    chip_stride: int,
    label_vector_file: Optional[PATH_TYPE] = None,
    label_column: Optional[str] = None,
    label_remap: Optional[dict] = None,
    write_empty_tiles: bool = False,
    drop_transparency: bool = True,
    remove_old: bool = True,
    output_suffix: str = ".JPG",
    ROI_file: Optional[PATH_TYPE] = None,
    background_ind: int = NULL_TEXTURE_INT_VALUE,
):
    """Take raster data and tile it for machine learning training or inference

    Args:
        raster_file (PATH_TYPE):
            Path to the raster file to tile.
        output_folder (PATH_TYPE):
            Where to write the tiled outputs.
        chip_size (int):
            Size of the square chip in pixels.
        chip_stride (int):
            The stride in pixels between sliding window tiles.
        label_vector_file (Optional[PATH_TYPE], optional):
            A path to a vector geofile for the same region as the raster file. If provided, a
            parellel folder structure will be written to the chipped images that contains the
            corresponding rasterized data from the vector file. This is primarily useful for
            generating training data for ML. Defaults to None.
        label_column (Optional[str], optional):
            Which column to use within the provided file. If not provided, the index will be used.
            Defaults to None.
        label_remap (Optional[dict], optional):
            A dictionary mapping from the values in the `label_column` to integers that will be used
            for rasterization. Defaults to None.
        write_empty_tiles (bool, optional):
            Should tiles with no vector data be written. Defaults to False.
        drop_transparency (bool, optional):
            Should the forth channel be dropped if present. Defaults to True.
        remove_old (bool, optional):
            Remove `output_folder` if present. Defaults to True.
        output_suffix (str, optional):
            Suffix for written imagery files. Defaults to ".JPG".
        ROI_file (Optional[PATH_TYPE], optional):
            Path to a geospatial region of interest to restrict tile generation to. Defaults to None.
        background_ind (int, optional):
            If labels are written, any un-labeled region will have this value.
            Defaults to `NULL_TEXTURE_INT_VALUE`.
    """
    # Remove the existing directory
    if remove_old and os.path.isdir(output_folder):
        shutil.rmtree(output_folder)

    # Read the labels if provided
    if label_vector_file is not None:
        label_gdf = gpd.read_file(label_vector_file)
    else:
        label_gdf = None

    # Open the raster file
    with rio.open(raster_file, "r") as dataset:
        working_CRS = dataset.crs
        # Create a list of windows for reading
        windows = create_windows(
            dataset_h_w=(dataset.height, dataset.width),
            window_size=chip_size,
            window_stride=chip_stride,
        )

        desc = f"Writing image chips to {output_folder}"
        if label_gdf is not None:
            desc = f"Writing image chips and labels to {output_folder}"
            label_gdf.to_crs(working_CRS, inplace=True)

            if label_column is not None:
                label_values = label_gdf[label_column].tolist()
            else:
                label_values = label_gdf.index.tolist()

            if label_remap is not None:
                label_values = [label_remap[old_label] for old_label in label_values]

            label_shapes = list(zip(label_gdf.geometry.values, label_values))
            labels_folder = Path(output_folder, "anns")
            output_folder = Path(output_folder, "imgs")

            ensure_folder(labels_folder)
        ensure_folder(output_folder)

        # Set up the ROI now that we have the working CRS
        if ROI_file is not None:
            ROI_gdf = gpd.read_file(ROI_file).to_crs(working_CRS)
            ROI_geometry = ROI_gdf.dissolve().geometry.values[0]
            if label_gdf is not None:
                # Crop the labels dataframe to the ROI
                label_gdf = label_gdf.intersection(ROI_geometry)
        else:
            ROI_geometry = None

        for window in tqdm(windows, desc=desc):
            if ROI_geometry is not None:
                window_transformer = AffineTransformer(dataset.window_transform(window))
                pixel_corners = (
                    (0, 0),
                    (0, chip_size),
                    (chip_size, chip_size),
                    (chip_size, 0),
                )
                geospatial_corners = [
                    window_transformer.xy(pc[0], pc[1], offset="ul")
                    for pc in pixel_corners
                ]
                geospatial_corners.append(geospatial_corners[0])
                window_polygon = Polygon(geospatial_corners)

                if not ROI_geometry.intersects(window_polygon):
                    # Skip writing this chip if it doesn't intersect the ROI
                    continue

            if label_gdf is not None:
                window_transform = dataset.window_transform(window)
                window_transformer = AffineTransformer(window_transform)
                labels_raster = rasterize(
                    label_shapes,
                    out_shape=(chip_size, chip_size),
                    transform=window_transform,
                    fill=background_ind,
                )
                labels_raster = labels_raster.astype(np.uint8)
                # See if we should skip this tile since it's only background data
                if not write_empty_tiles and np.all(
                    labels_raster == NULL_TEXTURE_INT_VALUE
                ):
                    continue

                # Write out the label
                output_file_name = Path(
                    labels_folder,
                    get_str_from_window(
                        raster_file=raster_file, window=window, suffix=".png"
                    ),
                )
                imwrite(
                    output_file_name,
                    pad_to_full_size(labels_raster, (chip_size, chip_size)),
                )

            windowed_raster = dataset.read(window=window)
            windowed_img = reshape_as_image(windowed_raster)

            if drop_transparency and windowed_img.shape[2] == 4:
                transparency = windowed_img[..., 3]
                windowed_img = windowed_img[..., :3]
                # Set transperent regions to black
                mask = transparency == 0
                if np.all(mask):
                    continue

                windowed_img[mask, :] = 0

            output_file_name = Path(
                output_folder,
                get_str_from_window(
                    raster_file=raster_file, window=window, suffix=output_suffix
                ),
            )
            imwrite(
                output_file_name,
                pad_to_full_size(
                    windowed_img,
                    (chip_size, chip_size),
                ),
            )


def assemble_tiled_predictions(
    raster_file: PATH_TYPE,
    pred_folder: PATH_TYPE,
    class_savefile: PATH_TYPE,
    num_classes: int,
    counts_savefile: Union[PATH_TYPE, None] = None,
    downweight_edge_frac: float = 0.25,
    nodataval: Union[int, None] = NULL_TEXTURE_INT_VALUE,
    count_dtype: type = np.uint8,
    max_overlapping_tiles: int = 4,
):
    """Take tiled predictions on disk and aggregate them into a raster

    Args:
        raster_file (PATH_TYPE):
            Path to the raster file used to generate chips. This is required only to understand the
            geospatial reference.
        pred_folder (PATH_TYPE):
            A folder where every file is a prediction for a tile. The filename must encode the
            bounds of the windowed crop.
        class_savefile (PATH_TYPE):
            Where to save the merged raster.
        counts_savefile (typing.Union[PATH_TYPE, NoneType], optional):
            Where to save the counts for the merged predictions raster.
            A tempfile will be created and then deleted if not specified. Defaults to None.
        downweight_edge_frac (float, optional):
            Downweight this fraction of predictions at the edge of each tile using a linear ramp. Defaults to 0.25.
        nodataval: (typing.Union[int, None]):
            Value for unassigned pixels. If None, will be set to num_classes, the first unused class. Defaults to None.
        count_dtype (type, optional):
            What type to use for aggregation. Float uses more space but is more accurate. Defaults to np.uint8
        max_overlapping_tiles (int):
            The max number of prediction tiles that may overlap at a given point. This is used to upper bound the valud in the count matrix,
            because we use scaled np.uint8 values rather than floats for efficiency. Setting a lower value enables slightly more accuracy in the
            aggregation process, but too low can lead to overflow. Defaults to 4
    """
    # Find the filenames of tiled predictions
    pred_files = [f for f in pred_folder.glob("*") if f.is_file()]

    # Set nodataval to the first unused class ID
    if nodataval is None:
        nodataval = num_classes

    # If the user didn't specify where to write the counts, create a tempfile that will be deleted
    if counts_savefile is None:
        # Create the containing folder if required
        ensure_containing_folder(class_savefile)
        counts_savefile_manager = tempfile.NamedTemporaryFile(
            mode="w+", suffix=".tif", dir=class_savefile.parent
        )
        counts_savefile = counts_savefile_manager.name

    # Parse the filenames to get the windows
    # TODO consider using the extent to only write a file for the minimum encolsing rectangle
    windows, extent = parse_windows_from_files(pred_files, return_in_extent_coords=True)

    # Aggregate predictions
    with rio.open(raster_file) as src:
        # Create file to store counts that is the same as the input raster except it has num_classes number of bands
        # TODO make this only the size of the extent computed by parse_windows_from_files
        extent_transform = src.window_transform(extent)

        with rio.open(
            counts_savefile,
            "w+",
            driver="GTiff",
            height=extent.height,
            width=extent.width,
            count=num_classes,
            dtype=count_dtype,
            crs=src.crs,
            transform=extent_transform,
        ) as dst:
            # Create
            pred_weighting_dict = {}
            for pred_file, window in tqdm(
                zip(pred_files, windows),
                desc="Aggregating raster predictions",
                total=len(pred_files),
            ):
                # Read the prediction from disk
                pred = read_image_or_numpy(pred_file)

                if pred.shape != (window.height, window.width):
                    raise ValueError("Size of pred does not match window")

                # We want to downweight portions at the edge so we create a ramped weighting mask
                # but we don't want to duplicate this computation because it's the same for each same sized chip
                if pred.shape not in pred_weighting_dict:
                    # We want to keep this as a uint8
                    pred_weighting = create_ramped_weighting(
                        pred.shape, downweight_edge_frac
                    )

                    # Allow us to get as much granularity as possible given the datatype
                    if count_dtype is not float:
                        pred_weighting = pred_weighting * (
                            np.iinfo(count_dtype).max / max_overlapping_tiles
                        )
                    # Convert weighting to desired type
                    pred_weighting_dict[pred.shape] = pred_weighting.astype(count_dtype)

                # Get weighting
                pred_weighting = pred_weighting_dict[pred.shape]

                # Update each band in the counts file within the window
                for i in range(num_classes):
                    # Bands in rasterio are 1-indexed
                    band_ind = i + 1
                    class_i_window_counts = dst.read(band_ind, window=window)
                    class_i_preds = pred == i
                    # If nothing matches this class, don't waste computation
                    if not np.any(class_i_preds):
                        continue
                    # Weight the predictions to downweight the ones at the edge
                    weighted_preds = (class_i_preds * pred_weighting).astype(
                        count_dtype
                    )
                    # Add the new predictions to the previous counts
                    class_i_window_counts += weighted_preds
                    # Write out the updated results for this window
                    dst.write(class_i_window_counts, band_ind, window=window)

    ## Convert counts file to max-class file

    with rio.open(counts_savefile, "r") as src:
        # Create a one-band file to store the index of the most predicted class
        with rio.open(
            class_savefile,
            "w",
            driver="GTiff",
            height=src.shape[0],
            width=src.shape[1],
            count=1,
            dtype=np.uint8,
            crs=src.crs,
            transform=src.transform,
            nodata=nodataval,
        ) as dst:
            # Iterate over the blocks corresponding to the tiff driver in the dataset
            # to compute the max class and write it out
            for _, window in tqdm(
                list(src.block_windows()), desc="Writing out max class"
            ):
                # Read in the counts
                counts_array = src.read(window=window)
                # Compute which pixels have no recorded predictions and mask them out
                nodata_mask = np.sum(counts_array, axis=0) == 0

                # If it's all nodata, don't write it out
                # TODO make sure this works as expected
                if np.all(nodata_mask):
                    continue

                # Compute which class had the highest counts
                max_class = np.argmax(counts_array, axis=0)
                max_class[nodata_mask] = nodataval
                # TODO, it would be good to check if it's all nodata and skip the write because that's unneeded
                dst.write(max_class, 1, window=window)
