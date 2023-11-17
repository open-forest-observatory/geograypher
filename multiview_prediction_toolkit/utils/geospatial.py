import logging
import typing

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pyproj
import rasterio as rio
from geopandas import GeoDataFrame
from rasterstats import zonal_stats
from shapely import MultiPolygon, intersection, union
from shapely.geometry import box
from tqdm import tqdm

from multiview_prediction_toolkit.config import PATH_TYPE


def ensure_geometric_CRS(geodata):
    if geodata.crs == pyproj.CRS.from_epsg(4326):
        point = geodata["geometry"][0].centroid
        geometric_crs = get_projected_CRS(lon=point.x, lat=point.y)
        return geodata.to_crs(geometric_crs)
    return geodata


def get_projected_CRS(lat, lon, assume_western_hem=True):
    if assume_western_hem and lon > 0:
        lon = -lon
    # https://gis.stackexchange.com/questions/190198/how-to-get-appropriate-crs-for-a-position-specified-in-lat-lon-coordinates
    epgs_code = 32700 - round((45 + lat) / 90) * 100 + round((183 + lon) / 6)
    crs = pyproj.CRS.from_epsg(epgs_code)
    return crs


def find_union_of_intersections(list_of_multipolygons, crs, vis=False):
    all_intersections = MultiPolygon()
    for i, multipolygon_a in enumerate(list_of_multipolygons):
        for multipolygon_b in list_of_multipolygons[:i]:
            new_intersection = intersection(multipolygon_a, multipolygon_b)
            all_intersections = union(all_intersections, new_intersection)
    if vis:
        geopandas_all_intersections = GeoDataFrame(
            geometry=[all_intersections], crs=crs
        )
        geopandas_all_intersections.plot()
        plt.show()
    return all_intersections


def get_overlap_raster(
    unlabeled_df: typing.Union[PATH_TYPE, GeoDataFrame],
    classes_raster: PATH_TYPE,
    num_classes: typing.Union[None, int] = None,
    normalize: bool = False,
) -> typing.Tuple[np.ndarray, np.ndarray]:
    """Get the overlap for each polygon in the unlabeled DF with each class in the raster

    Args:
        unlabeled_df (typing.Union[PATH_TYPE, GeoDataFrame]):
            Dataframe or path to dataframe containing geometries per object
        classes_raster (PATH_TYPE): Path to a categorical raster
        num_classes (typing.Union[None, int], optional):
            Number of classes, if None defaults to the highest overlapping class. Defaults to None.
        normalize (bool, optional): Normalize counts matrix from pixels to fraction. Defaults to False.

    Returns:
        np.ndarray: (n_valid, n_classes) counts per polygon per class
        np.ndarray: (n_valid,) indices into the original array for polygons with non-null predictions
    """
    # Try to load the vector data if it's not a geodataframe
    if not isinstance(unlabeled_df, GeoDataFrame):
        unlabeled_df = gpd.read_file(unlabeled_df)

    with rio.open(classes_raster, "r") as src:
        raster_crs = src.crs
        # Note this is a shapely object with no CRS, but we ensure
        # the vectors are converted to the same CRS
        raster_bounds = box(*src.bounds)

    # Ensure that the vector data is in the same CRS as the raster
    if raster_crs != unlabeled_df.crs:
        # Avoid doing this in place because we don't want to modify the input dataframe
        # This should properly create a copy
        unlabeled_df = unlabeled_df.to_crs(raster_crs)
        print(unlabeled_df.crs)

    # Find the polygons that are within the bounds of the raster
    within_bounds_IDs = np.where(
        np.logical_not(unlabeled_df.intersection(raster_bounds).is_empty.to_numpy())
    )[0]

    # Compute the stats
    stats = zonal_stats(
        unlabeled_df.iloc[within_bounds_IDs], str(classes_raster), categorical=True
    )

    # Find which polygons have non-null class predictions
    # Due to nondata regions, some polygons within the region may not have class information
    valid_prediction_IDs = np.where([x != {} for x in stats])[0]

    # Determine the number of classes if not set
    if num_classes is None:
        # Find the max value that show up in valid predictions
        num_classes = 1 + np.max(
            [np.max(list(stats[i].keys())) for i in valid_prediction_IDs]
        )

    # Build the counts matrix for non-null predictions
    counts_matrix = np.zeros((len(valid_prediction_IDs), num_classes))

    # Fill the counts matrix
    for i in valid_prediction_IDs:
        for j, count in stats[i].items():
            counts_matrix[i, j] = count

    # Bookkeeping to find the IDs that were both within the raster and non-null
    valid_IDs_in_original = within_bounds_IDs[valid_prediction_IDs]

    return counts_matrix, valid_IDs_in_original


# https://gis.stackexchange.com/questions/421888/getting-the-percentage-of-how-much-areas-intersects-with-another-using-geopandas
def get_fractional_overlap_vector(
    unlabeled_df: GeoDataFrame, classes_df: GeoDataFrame, class_column: str = "names"
) -> GeoDataFrame:
    """
    For each element in unlabeled df, return the fractional overlap with each class in
    classes_df


    Args:
        unlabeled_df (GeoDataFrame): A dataframe of geometries
        classes_df (GeoDataFrame): A dataframe of classes
        class_column (str, optional): Which column in the classes_df to use. Defaults to "names".

    Returns:
        GeoDataFrame: A multi-level dataframe for each element in unlabeled_df that
                      overlaps with the classe. The second level is the overlap with
                      each class
    """
    # Find the union of all class information
    union_of_all_classes = classes_df.dissolve()

    # This column will be used later to index back into the original dataset
    unlabeled_df["index"] = unlabeled_df.index
    # Find all the polygons intersecting the class data
    unlabeled_polygons_intersecting_classes = union_of_all_classes.overlay(
        unlabeled_df, how="intersection"
    )
    # We can't use the intersecting polygons directly because we want to preserve full geometries at the boundaries
    intersecting_indices = unlabeled_polygons_intersecting_classes["index"].to_numpy()
    # Find the subset of original polygons that overlap with the class data
    unlabeled_df_intersecting_classes = unlabeled_df.iloc[intersecting_indices]

    # Add area field to each
    unlabeled_df_intersecting_classes[
        "unlabeled_area"
    ] = unlabeled_df_intersecting_classes.area

    # Find the intersecting geometries
    # We want only the ones that have some overlap with the unlabeled geometry, but I don't think that can be specified
    overlay = gpd.overlay(
        unlabeled_df_intersecting_classes,
        classes_df,
        how="union",
        keep_geom_type=False,
    )
    # Drop the rows that only contain information from the class_labels
    overlay = overlay[np.isfinite(overlay["index"].to_numpy())]

    overlay["overlapping_area"] = overlay.area
    overlay["per_class_area_fraction"] = (
        overlay["overlapping_area"] / overlay["unlabeled_area"]
    )

    # Aggregating the results
    results = overlay.groupby(["index", class_column]).agg(
        {"per_class_area_fraction": "sum"}
    )

    # Set the max class
    argmax = results.groupby(level=[0]).idxmax()
    max_class = [x[1] for x in argmax.iloc[:, 0].to_list()]
    index = [int(x[0]) for x in argmax.iloc[:, 0].to_list()]
    unlabeled_df_intersecting_classes.loc[index, "predicted_class"] = max_class

    return results, overlay, unlabeled_df_intersecting_classes


# https://stackoverflow.com/questions/60288953/how-to-change-the-crs-of-a-raster-with-rasterio
def reproject_raster(in_path, out_path, out_crs=pyproj.CRS.from_epsg(4326)):
    """ """
    logging.warn("Starting to reproject raster")
    # reproject raster to project crs
    with rio.open(in_path) as src:
        src_crs = src.crs
        transform, width, height = rio.warp.calculate_default_transform(
            src_crs, out_crs, src.width, src.height, *src.bounds
        )
        kwargs = src.meta.copy()

        kwargs.update(
            {"crs": out_crs, "transform": transform, "width": width, "height": height}
        )

        with rio.open(out_path, "w", **kwargs) as dst:
            for i in tqdm(range(1, src.count + 1), desc="Reprojecting bands"):
                rio.warp.reproject(
                    source=rio.band(src, i),
                    destination=rio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=out_crs,
                    resampling=rio.warp.Resampling.nearest,
                )
    logging.warn("Done reprojecting raster")
