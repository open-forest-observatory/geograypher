import typing
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio as rio
from IPython.core.debugger import set_trace
from rasterio.plot import reshape_as_image
from rasterstats import zonal_stats
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from geograypher.constants import (
    CLASS_ID_KEY,
    CLASS_NAMES_KEY,
    PATH_TYPE,
    TEN_CLASS_VIS_KWARGS,
)
from geograypher.utils.files import ensure_containing_folder
from geograypher.utils.geospatial import (
    coerce_to_geoframe,
    ensure_projected_CRS,
    get_overlap_raster,
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
    class_column=CLASS_NAMES_KEY,
    ignore_class=255,
    vis_kwargs=TEN_CLASS_VIS_KWARGS,
):
    if "clim" in vis_kwargs:
        vmin, vmax = vis_kwargs["clim"]
    else:
        vmin, vmax = (None, None)

    cmap = vis_kwargs.pop("cmap", None)
    if check_if_raster(filename):
        with rio.open(filename) as dataset:
            # resample data to target shape
            raster = dataset.read(
                out_shape=(
                    dataset.count,
                    int(dataset.height * raster_downsample_factor),
                    int(dataset.width * raster_downsample_factor),
                ),
                resampling=rio.enums.Resampling.bilinear,
            )

        single_channel = raster.shape[0] == 1

        if single_channel:
            img = np.squeeze(raster).astype(float)
            img[img == ignore_class] = np.nan
            interpolation = "none"
        else:
            img = reshape_as_image(raster)
            # Auto brighten if dark
            # walrus operator to avoid re-computing the mean
            if img.shape[2] == 3:
                mean_img = np.mean(img)
            elif img.shape[2] == 4:
                mean_img = np.mean(img[img[..., 3] > 0, :3])
            else:
                raise ValueError()

            if mean_img < 50:
                img = np.clip((img * (50 / mean_img)), 0, 255).astype(np.uint8)

            interpolation = "antialiased"

        cb = ax.imshow(
            img, vmin=vmin, vmax=vmax, cmap=cmap, interpolation=interpolation
        )
        if single_channel:
            plt.colorbar(cb)
    else:
        data = gpd.read_file(filename)
        data.plot(class_column, ax=ax, vmin=vmin, vmax=vmax, cmap=cmap)


def cf_from_vector_vector(
    predicted_df: gpd.GeoDataFrame,
    true_df: gpd.GeoDataFrame,
    column_name: str,
    column_values: list[str] = None,
    include_unlabeled_class: bool = True,
):
    if not isinstance(predicted_df, gpd.GeoDataFrame):
        predicted_df = gpd.read_file(predicted_df)
    if not isinstance(true_df, gpd.GeoDataFrame):
        true_df = gpd.read_file(true_df)

    grouped_predicted_df = predicted_df.dissolve(by=column_name)
    grouped_true_df = true_df.dissolve(by=column_name)

    if column_values is None:
        column_values = grouped_true_df.index.tolist()

    grouped_predicted_df = ensure_projected_CRS(grouped_predicted_df)
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

    if include_unlabeled_class:
        # Compute the sum of labeled predictions per ground truth class
        labeled_areas = np.sum(confusion_matrix, axis=1)
        # Compute the area of each ground truth class
        total_areas = grouped_true_df.area.values
        # TODO this isn't correct if predictions are overlapping
        unlabeled_areas = total_areas - labeled_areas

        # Create a cf matrix with an un-labeled class
        confusion_matrix_w_unlabeled = np.zeros(
            (confusion_matrix.shape[0] + 1, confusion_matrix.shape[1] + 1)
        )
        confusion_matrix_w_unlabeled[:-1, :-1] = confusion_matrix
        confusion_matrix_w_unlabeled[:-1, -1] = unlabeled_areas
        confusion_matrix = confusion_matrix_w_unlabeled
    return confusion_matrix, column_values


def compute_confusion_matrix_from_geospatial(
    prediction_file: PATH_TYPE,
    groundtruth_file: PATH_TYPE,
    class_names: list[str],
    vis_raster_file: typing.Union[PATH_TYPE, None] = None,
    vis_savefile: str = None,
    vis: bool = True,
    normalize: bool = True,
    remap_raster_inds=None,
    column_name=CLASS_NAMES_KEY,
):
    pred_is_raster = check_if_raster(prediction_file)
    gt_is_raster = check_if_raster(groundtruth_file)

    if vis:
        has_vis_raster_file = int(vis_raster_file is not None)
        _, axs = plt.subplots(1, 2 + has_vis_raster_file)

        if has_vis_raster_file > 0:
            plot_geodata(vis_raster_file, ax=axs[0], vis_kwargs={})
            axs[0].set_title("Reference\n raster file")

        plot_geodata(
            prediction_file,
            ax=axs[0 + has_vis_raster_file],
            class_column=column_name,
        )
        plot_geodata(
            groundtruth_file,
            ax=axs[1 + has_vis_raster_file],
            class_column=column_name,
        )

        axs[0 + has_vis_raster_file].set_title("Predictions")
        axs[1 + has_vis_raster_file].set_title("Ground truth")
        # TODO make this more flexible
        plt.show()

    if gt_is_raster:
        raise NotImplementedError()
    else:
        groundtruth_gdf = coerce_to_geoframe(groundtruth_file)
        groundtruth_gdf = groundtruth_gdf.set_index(CLASS_NAMES_KEY, drop=True).reindex(
            class_names
        )

        if pred_is_raster:
            ret = zonal_stats(groundtruth_gdf, str(prediction_file), categorical=True)

            cf_matrix = np.array([list(x.values()) for x in ret])
            if remap_raster_inds is not None:
                cf_matrix = cf_matrix[:, tuple(remap_raster_inds)]
            cf_matrix = np.pad(cf_matrix, pad_width=((0, 1), (0, 1)))
            classes = class_names
        else:
            prediction_gdf = coerce_to_geoframe(prediction_file)
            prediction_gdf = prediction_gdf.set_index(
                CLASS_NAMES_KEY, drop=True
            ).reindex(class_names)

            cf_matrix, classes = cf_from_vector_vector(
                predicted_df=prediction_gdf,
                true_df=groundtruth_gdf,
                column_name=column_name,
            )

        class_names.append("unlabeled")

    if normalize:
        cf_matrix = cf_matrix / cf_matrix.sum()

    conf_disp = ConfusionMatrixDisplay(cf_matrix, display_labels=class_names)
    conf_disp.plot()

    if vis_savefile is None:
        plt.show()
    else:
        plt.savefig(vis_savefile)
        plt.close()

    accuracy = np.sum(cf_matrix * np.eye(cf_matrix.shape[0])) / np.sum(cf_matrix)

    return cf_matrix, classes, accuracy


def compute_and_show_cf(
    pred_labels: list,
    gt_labels: list,
    labels: typing.Union[None, typing.List[str]] = None,
    use_labels_from: str = "both",
    vis: bool = True,
    cf_plot_savefile: typing.Union[None, PATH_TYPE] = None,
    cf_np_savefile: typing.Union[None, PATH_TYPE] = None,
):
    """_summary_

    Args:
        pred_labels (list): _description_
        gt_labels (list): _description_
        labels (typing.Union[None, typing.List[str]], optional): _description_. Defaults to None.
        use_labels_from (str, optional): _description_. Defaults to "both".
        vis (bool, optional): _description_. Defaults to True.
        cf_plot_savefile (typing.Union[None, PATH_TYPE], optional): _description_. Defaults to None.
        cf_np_savefile (typing.Union[None, PATH_TYPE], optional): _description_. Defaults to None.

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    if labels is None:
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
        if cf_plot_savefile is None:
            plt.show()
        else:
            ensure_containing_folder(cf_plot_savefile)
            plt.savefig(cf_plot_savefile)

    if cf_np_savefile:
        ensure_containing_folder(cf_np_savefile)
        np.save(cf_np_savefile, cf_matrix)

    # TODO compute more comprehensive metrics here
    accuracy = np.sum(cf_matrix * np.eye(cf_matrix.shape[0])) / np.sum(cf_matrix)

    return cf_matrix, labels, accuracy


def compute_comprehensive_metrics(cf_matrix: np.ndarray, class_names: typing.List[str]):
    accuracy = np.sum(cf_matrix * np.eye(cf_matrix.shape[0])) / np.sum(cf_matrix)

    num_classes = cf_matrix.shape[0]

    per_class_dict = {}

    for i, class_name in zip(range(num_classes), class_names):
        total = np.sum(cf_matrix)
        true_positives = cf_matrix[i, i]
        num_true = np.sum(cf_matrix[i, :])
        num_pred = np.sum(cf_matrix[:, i])
        recall = true_positives / num_true
        precision = true_positives / num_pred

        true_neg = total + true_positives - num_true - num_pred
        acc = (true_positives + true_neg) / np.sum(cf_matrix)

        per_class_dict[class_name] = {
            "acc": acc,
            "recall": recall,
            "precision": precision,
            "num_true": num_true,
            "num_pred": num_pred,
        }

    precisions = np.array([v["precision"] for v in per_class_dict.values()])
    recalls = np.array([v["recall"] for v in per_class_dict.values()])
    num_trues = np.array([v["num_true"] for v in per_class_dict.values()])

    precisions = np.nan_to_num(precisions)

    any_trues = num_trues > 0

    class_averaged_recall = np.mean(recalls[any_trues])
    class_averaged_precision = np.mean(precisions[any_trues])

    return {
        "accuracy": accuracy,
        "class_averaged_recall": class_averaged_recall,
        "class_averaged_precision": class_averaged_precision,
        "per_class": per_class_dict,
    }
