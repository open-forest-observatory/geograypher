import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pyproj

from multiview_prediction_toolkit.config import PATH_TYPE
from multiview_prediction_toolkit.utils.geospatial import get_projected_CRS


def eval_confusion_matrix(
    predicted_df: gpd.GeoDataFrame,
    true_df: gpd.GeoDataFrame,
    column_name: str,
    column_values: list[str] = None,
    normalize: bool = True,
    normalize_by_class: bool = False,
    savepath: PATH_TYPE = None,
):
    grouped_predicted_df = predicted_df.dissolve(by=column_name)
    grouped_true_df = true_df.dissolve(by=column_name)

    if column_values is None:
        column_values = grouped_true_df.index.tolist()

    crs = grouped_predicted_df.crs
    if crs == pyproj.CRS.from_epsg(4326):
        centroid = grouped_predicted_df["geometry"][0].centroid
        crs = get_projected_CRS(lat=centroid.y, lon=centroid.x)

    grouped_predicted_df.to_crs(crs, inplace=True)
    grouped_true_df.to_crs(crs, inplace=True)

    confusion_matrix = np.zeros((len(column_values), len(column_values)))

    for i, true_val in enumerate(column_values):
        for j, pred_val in enumerate(column_values):
            pred_multipolygon = grouped_predicted_df.loc[
                grouped_predicted_df.index == pred_val
            ]["geometry"].values[0]
            true_multipolygon = grouped_true_df.loc[grouped_true_df.index == true_val][
                "geometry"
            ].values[0]
            intersection_area = pred_multipolygon.intersection(true_multipolygon).area
            confusion_matrix[i, j] = intersection_area

    if normalize:
        confusion_matrix = confusion_matrix / np.sum(confusion_matrix)

    if normalize_by_class:
        class_freq = np.sum(confusion_matrix, axis=1, keepdims=True)
        confusion_matrix = confusion_matrix / class_freq

    plt.imshow(confusion_matrix, vmin=0)
    plt.xticks(ticks=np.arange(len(column_values)), labels=column_values, rotation=45)
    plt.yticks(ticks=np.arange(len(column_values)), labels=column_values)
    plt.colorbar()
    plt.title("Confusion matrix")
    plt.xlabel("Predicted classes")
    plt.ylabel("True classes")

    if savepath is None:
        plt.show()
    else:
        plt.tight_layout()
        plt.savefig(savepath)
        plt.close()

    return confusion_matrix, column_values
