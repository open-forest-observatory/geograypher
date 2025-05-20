import logging
import typing

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pyproj
import rasterio as rio
from geopandas import GeoDataFrame, GeoSeries
from IPython.core.debugger import set_trace
from rasterstats import zonal_stats
from shapely import (
    Geometry,
    MultiPolygon,
    Polygon,
    difference,
    intersection,
    make_valid,
    union,
)
from shapely.geometry import box
from tqdm import tqdm

from geograypher.constants import LAT_LON_CRS, PATH_TYPE


def ensure_projected_CRS(geodata: gpd.GeoDataFrame):
    """Returns a projected geodataframe from the provided geodataframe by converting it to
    ESPG:4326 (if not already) and determining the projected CRS from the point
    coordinates.

    Args:
        geodata (gpd.GeoDataGrame): Original geodataframe that is potentially unprojected
    Returns:
        gpd.GeoDataGrame: projected geodataframe
    """
    # If CRS is projected return immediately
    if geodata.crs.is_projected:
        return geodata

    # If CRS is geographic and not long-lat, convert it to long-lat
    if geodata.crs.is_geographic and geodata.crs != LAT_LON_CRS:
        geodata = geodata.to_crs(LAT_LON_CRS)

    # Convert geographic long-lat CRS to projected CRS
    point = geodata["geometry"][0].centroid
    geometric_crs = get_projected_CRS(lon=point.x, lat=point.y)
    return geodata.to_crs(geometric_crs)


def get_projected_CRS(lat, lon, assume_western_hem=True):
    if assume_western_hem and lon > 0:
        lon = -lon
    # https://gis.stackexchange.com/questions/190198/how-to-get-appropriate-crs-for-a-position-specified-in-lat-lon-coordinates
    epgs_code = 32700 - round((45 + lat) / 90) * 100 + round((183 + lon) / 6)
    crs = pyproj.CRS.from_epsg(epgs_code)
    return crs


def convert_CRS_3D_points(points, input_CRS, output_CRS):
    transformer = pyproj.Transformer.from_crs(input_CRS, output_CRS)

    # Transform the coordinates
    points_in_output_CRS = transformer.transform(
        xx=points[:, 0],
        yy=points[:, 1],
        zz=points[:, 2],
    )
    # Stack and transpose
    points_in_output_CRS = np.vstack(points_in_output_CRS).T
    return points_in_output_CRS


def ensure_non_overlapping_polygons(
    geometries: typing.Union[typing.List[Geometry], gpd.GeoDataFrame],
    inplace: bool = False,
):
    # Make sure geometries is a list of shapely objects
    if isinstance(geometries, gpd.GeoDataFrame):
        original_gdf = geometries
        geometries = geometries.geometry.tolist()
    else:
        original_gdf = None

    output_geometries = [None] * len(geometries)
    union_of_added_geoms = MultiPolygon()

    areas = [geom.area for geom in geometries]
    sorted_inds = np.argsort(areas)

    for ind in tqdm(sorted_inds):
        # Get the input geometry and ensure it's valid
        geom = make_valid(geometries[ind])
        # Subtract the union of all
        geom_to_add = difference(geom, union_of_added_geoms)
        output_geometries[ind] = geom_to_add
        # Add the original geom, not the difference'd one, to avoid boundary artifacts
        union_of_added_geoms = union(geom, union_of_added_geoms)

    if original_gdf is None:
        return output_geometries
    elif inplace:
        original_gdf.geometry = output_geometries
    else:
        output_gdf = original_gdf.copy()
        output_gdf.geometry = output_geometries
        return output_gdf


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


def intersects_union_of_polygons(
    query_polygons: GeoDataFrame,
    region_polygon: typing.Union[GeoDataFrame, GeoSeries, Polygon, MultiPolygon],
):
    if isinstance(region_polygon, GeoDataFrame):
        region_polygon.plot()
        # Try to make geometries valid
        region_polygon.geometry = region_polygon.buffer(0)
        region_polygon = region_polygon.dissolve()
        region_polygon = region_polygon.geometry[0]

    # Find the polygons that are within the bounds of the raster
    intersection = query_polygons.intersection(region_polygon)
    empty_geometry = intersection.is_empty.to_numpy()
    within_bounds_IDs = np.where(np.logical_not(empty_geometry))[0]
    return within_bounds_IDs


def coerce_to_geoframe(potential_geoframe):
    # Try to load the vector data if it's not a geodataframe
    if not isinstance(potential_geoframe, GeoDataFrame):
        potential_geoframe = gpd.read_file(potential_geoframe)
    return potential_geoframe


def get_overlap_raster(
    unlabeled_df: typing.Union[PATH_TYPE, GeoDataFrame],
    classes_raster: PATH_TYPE,
    num_classes: typing.Union[None, int] = None,
    normalize: bool = False,
) -> (np.ndarray, np.ndarray):
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
    unlabeled_df = coerce_to_geoframe(unlabeled_df)

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

    # Compute which polygons intersect the raster region
    within_bounds_IDs = intersects_union_of_polygons(unlabeled_df, raster_bounds)

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

    if normalize:
        counts_matrix = counts_matrix / np.sum(counts_matrix, axis=1, keepdims=True)

    return counts_matrix, valid_IDs_in_original


# https://gis.stackexchange.com/questions/421888/getting-the-percentage-of-how-much-areas-intersects-with-another-using-geopandas
def get_overlap_vector(
    unlabeled_df: GeoDataFrame,
    classes_df: GeoDataFrame,
    class_column: str,
    normalize: bool = False,
) -> (np.ndarray, np.ndarray):
    """
    For each element in unlabeled df, return the fractional overlap with each class in
    classes_df


    Args:
        unlabeled_df (GeoDataFrame): A dataframe of geometries
        classes_df (GeoDataFrame): A dataframe of classes
        class_column (str, optional): Which column in the classes_df to use. Defaults to "names".
        normalize (bool, optional): Normalize counts matrix from area to fraction. Defaults to False.

    Returns:
        np.ndarray: (n_valid, n_classes) counts per polygon per class
        np.ndarray: (n_valid,) indices into the original array for polygons with non-null predictions
    """

    ## Preprocessing
    # Ensure that both a geodataframes
    unlabeled_df = coerce_to_geoframe(unlabeled_df)
    classes_df = coerce_to_geoframe(classes_df)

    unlabeled_df = ensure_projected_CRS(unlabeled_df)
    if classes_df.crs != unlabeled_df.crs:
        classes_df = classes_df.to_crs(unlabeled_df.crs)

    unlabeled_df.geometry = unlabeled_df.geometry.simplify(0.01)
    classes_df.geometry = classes_df.geometry.simplify(0.01)

    if class_column not in classes_df.columns:
        raise ValueError(f"Class column `{class_column}` not in {classes_df.columns}")

    logging.info(
        "Computing the intersection of the unlabeled polygons with the labeled region"
    )
    # Find which unlabeled polygons intersect with the labeled region
    intersection_IDs = intersects_union_of_polygons(unlabeled_df, classes_df)
    logging.info("Finished computing intersection")
    # Extract only these polygons
    unlabeled_df_intersecting_classes = unlabeled_df.iloc[intersection_IDs]
    unlabeled_df_intersecting_classes["index"] = unlabeled_df_intersecting_classes.index

    # Add area field to each
    unlabeled_df_intersecting_classes["unlabeled_area"] = (
        unlabeled_df_intersecting_classes.area
    )

    # Find the intersecting geometries
    # We want only the ones that have some overlap with the unlabeled geometry, but I don't think that can be specified
    logging.info("computing overlay")
    overlay = gpd.overlay(
        classes_df,
        unlabeled_df_intersecting_classes,
        how="union",
        keep_geom_type=False,
    )
    # Drop the rows that only contain information from the class_labels
    overlay = overlay.dropna(subset="index")

    # TODO look more into this part, something seems wrong
    overlay["overlapping_area"] = overlay.area
    # overlay["per_class_area_fraction"] = (
    #    overlay["overlapping_area"] / overlay["unlabeled_area"]
    # )
    # Aggregating the results
    # WARNING Make sure that this is a list and not a tuple or it gets considered one key
    logging.info("computing groupby")

    # If the two dataframes have a column with the same name, they will be renamed
    if (
        f"{class_column}_1" in overlay.columns
        and f"{class_column}_2" in overlay.columns
    ):
        aggregatation_class_column = f"{class_column}_1"
    else:
        aggregatation_class_column = class_column

    # Groupby and aggregate
    grouped_by = overlay.groupby(by=["index", aggregatation_class_column])
    aggregated = grouped_by.agg({"overlapping_area": "sum"})

    # Extract the original class names
    unique_class_names = sorted(classes_df[class_column].unique().tolist())
    # And the indices from the original dataframe. This is relavent if the input
    # dataframe was a subset of an original one
    unique_index_values = sorted(unlabeled_df_intersecting_classes.index.tolist())

    counts_matrix = np.zeros(
        (len(unlabeled_df_intersecting_classes), len(unique_class_names))
    )

    for r in aggregated.iterrows():
        (index, class_name), area = r
        # The index is the index from the original unlabeled dataset, but we need the index into the subset
        unlabled_object_ind = unique_index_values.index(index)
        # Transform the class name into a class index
        class_ind = unique_class_names.index(class_name)
        # Set the value to the area of the overlap between that unlabeled object and given class
        counts_matrix[unlabled_object_ind, class_ind] = float(area.iloc[0])

    if normalize:
        counts_matrix = counts_matrix / np.sum(counts_matrix, axis=1, keepdims=True)

    return counts_matrix, intersection_IDs, unique_class_names


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


def load_downsampled_raster_data(dataset_filename: PATH_TYPE, downsample_factor: float):
    """Load a raster file spatially downsampled

    Args:
        dataset (PATH_TYPE): Path to the raster
        downsample_factor (float): Downsample factor of 10 means that pixels are 10 times larger

    Returns:
        np.array: The downsampled array in the rasterio (c, h, w) convention
        rio.DatasetReader: The reader with the transform updated
        rio.Transform: The updated transform
    """
    # Open the dataset handler. Note that this doesn't read into memory.
    dataset = rio.open(dataset_filename)

    # resample data to target shape
    data = dataset.read(
        out_shape=(
            dataset.count,
            int(dataset.height / downsample_factor),
            int(dataset.width / downsample_factor),
        ),
        resampling=rio.enums.Resampling.bilinear,
    )

    # scale image transform
    updated_transform = dataset.transform * dataset.transform.scale(
        (dataset.width / data.shape[-1]), (dataset.height / data.shape[-2])
    )
    # Return the data and the transform
    return data, dataset, updated_transform
