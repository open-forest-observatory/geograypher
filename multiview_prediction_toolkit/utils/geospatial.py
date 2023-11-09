import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pyproj
from geopandas import GeoDataFrame
from shapely import MultiPolygon, intersection, union


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


def to_float():
    pass


# https://gis.stackexchange.com/questions/421888/getting-the-percentage-of-how-much-areas-intersects-with-another-using-geopandas
def get_fractional_overlap(
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
