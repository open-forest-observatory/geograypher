import typing
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio as rio
from IPython.core.debugger import set_trace
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from multiview_mapping_toolkit.constants import PATH_TYPE, TEN_CLASS_VIS_KWARGS
from multiview_mapping_toolkit.utils.geospatial import (
    ensure_geometric_CRS,
    get_overlap_raster,
    get_overlap_vector,
)


def check_if_raster(filename):
    extension = Path(filename).suffix
    if extension.lower() in (".tif", ".tiff"):
        return True
    elif extension.lower() in (".geojson", ".shp"):
        return False
    else:
        raise ValueError("Unknown extension")


def plot_geodata(
    filename,
    ax,
    raster_downsample_factor=0.1,
    class_column="class_id",
    ignore_class=255,
    vis_kwargs=TEN_CLASS_VIS_KWARGS,
):
    vmin, vmax = vis_kwargs["clim"]
    if check_if_raster(filename):
        with rio.open(filename) as dataset:
            # resample data to target shape
            data = dataset.read(
                out_shape=(
                    dataset.count,
                    int(dataset.height * raster_downsample_factor),
                    int(dataset.width * raster_downsample_factor),
                ),
                resampling=rio.enums.Resampling.bilinear,
            )
        data = data[0].astype(float)
        data[data == ignore_class] = np.nan
        plt.colorbar(
            ax.imshow(data, vmin=vmin, vmax=vmax, cmap=vis_kwargs["cmap"]), ax=ax
        )
    else:
        data = gpd.read_file(filename)
        data.plot(class_column, ax=ax, vmin=vmin, vmax=vmax, cmap=vis_kwargs["cmap"])


def cf_from_vector_vector(
    predicted_df: gpd.GeoDataFrame,
    true_df: gpd.GeoDataFrame,
    column_name: str,
    column_values: list[str] = None,
):
    if not isinstance(predicted_df, gpd.GeoDataFrame):
        predicted_df = gpd.read_file(predicted_df)
    if not isinstance(true_df, gpd.GeoDataFrame):
        true_df = gpd.read_file(true_df)

    grouped_predicted_df = predicted_df.dissolve(by=column_name)
    grouped_true_df = true_df.dissolve(by=column_name)

    if column_values is None:
        column_values = grouped_true_df.index.tolist()

    grouped_predicted_df = ensure_geometric_CRS(grouped_predicted_df)
    grouped_true_df.to_crs(grouped_predicted_df.crs, inplace=True)

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

    return confusion_matrix, column_values


def compute_confusion_matrix_from_geospatial(
    prediction_file: PATH_TYPE,
    groundtruth_file: PATH_TYPE,
    class_names: list[str],
    vis_savefile: str = None,
    normalize: bool = True,
    column_name=None,
):
    pred_is_raster = check_if_raster(prediction_file)
    gt_is_raster = check_if_raster(groundtruth_file)

    if gt_is_raster and pred_is_raster:
        raise NotImplementedError()
    elif not gt_is_raster and pred_is_raster:
        cf_matrix, classes = get_overlap_raster(groundtruth_file, prediction_file)
    elif gt_is_raster and not pred_is_raster:
        # TODO use get_overlap_raster and flip the order, but make sure to properly threshold area
        raise NotImplementedError()
    elif not gt_is_raster and not pred_is_raster:
        cf_matrix, classes = get_overlap_vector(
            unlabeled_df=groundtruth_file,
            classes_df=prediction_file,
            class_column=column_name,
        )

    if normalize:
        cf_matrix /= cf_matrix.sum()

    conf_disp = ConfusionMatrixDisplay(cf_matrix, display_labels=class_names)
    conf_disp.plot()

    if vis_savefile is None:
        plt.show()
    else:
        plt.savefig(vis_savefile)
        plt.close()

    accuracy = np.sum(cf_matrix * np.eye(cf_matrix.shape[0]))

    return cf_matrix, classes, accuracy


def compute_and_show_cf(
    pred_labels: list,
    gt_labels: list,
    use_labels_from: str = "both",
    vis: bool = True,
    savefile: typing.Union[None, PATH_TYPE] = None,
):
    """_summary_

    Args:
        pred_labels (list): _description_
        gt_labels (list): _description_
        use_labels_from (str, optional): _description_. Defaults to "gt".
    """
    if use_labels_from == "gt":
        labels = np.unique(list(gt_labels))
    elif use_labels_from == "pred":
        labels = np.unique(list(pred_labels))
    elif use_labels_from == "both":
        labels = np.unique(list(pred_labels) + list(gt_labels))
    else:
        raise ValueError(
            f"Must use labels from gt, pred, or both but instead was {use_labels_from}"
        )

    cf_matrix = confusion_matrix(y_true=gt_labels, y_pred=pred_labels, labels=labels)

    if vis:
        cf_disp = ConfusionMatrixDisplay(
            confusion_matrix=cf_matrix, display_labels=labels
        )
        cf_disp.plot()
        if savefile is None:
            plt.show()
        else:
            plt.savefig(savefile)

    # TODO compute more comprehensive metrics here
    accuracy = np.sum(cf_matrix * np.eye(cf_matrix.shape[0])) / np.sum(cf_matrix)

    return cf_matrix, labels, accuracy
