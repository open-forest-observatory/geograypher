import logging
import tempfile
import typing
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pyproj
import rasterio as rio
from rastervision.core.data import ClassConfig
from rastervision.core.data.utils import make_ss_scene
from rastervision.core.evaluation import SemanticSegmentationEvaluator
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from multiview_prediction_toolkit.config import PATH_TYPE
from multiview_prediction_toolkit.utils.geospatial import (
    get_projected_CRS,
    reproject_raster,
)


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


def check_if_raster(filename):
    extension = Path(filename).suffix
    if extension.lower() in (".tif", ".tiff"):
        return True
    elif extension.lower() in (".geojson", ".shp"):
        return False
    else:
        raise ValueError("Unknown extension")


def plot_geodata(filename, ax, raster_downsample_factor=0.1, ignore_class=3):
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
        ax.imshow(data)
    else:
        data = gpd.read_file(filename)
        data.plot("class_id", ax=ax)


def make_ss_scene_vec_or_rast(
    class_config, image_file, label_file, validate_vector_label=True
):
    image_data = rio.open(image_file)

    if image_data.crs != pyproj.CRS.from_epsg(4326):
        temp_image_file = tempfile.NamedTemporaryFile(mode="w+", suffix=".tif")
        reproject_raster(image_file, temp_image_file.name)
        image_file = temp_image_file.name

    kwargs = {"class_config": class_config, "image_uri": image_file}
    is_raster_label = check_if_raster(label_file)

    temp_label_file_manager = None

    if is_raster_label:
        kwargs["label_raster_uri"] = label_file
        kwargs["label_raster_source_kw"] = dict(allow_streaming=True)
    else:
        if validate_vector_label:
            label_data = gpd.read_file(label_file)
            if "class_id" not in label_data.columns:
                if len(label_data) != len(class_config.names) - 1:
                    # TODO see if "class" is there and create labels based on the class names
                    raise ValueError(
                        "class_id not set and the number of rows is not the same as the number of classes"
                    )
                label_data["class_id"] = label_data.index
                # Create a temp file where we add the "class_id" field
                temp_label_file_manager = tempfile.NamedTemporaryFile(
                    mode="w+", suffix=".geojson"
                )
                label_file = temp_label_file_manager.name
                logging.warn(f"Created temp file {label_file}")
                # Dump data to tempfile
                label_data.to_file(label_file)
            if label_data.crs != pyproj.CRS.from_epsg(4326):
                temp_label_file_manager = tempfile.NamedTemporaryFile(
                    mode="w+", suffix=".geojson"
                )
                label_file = temp_label_file_manager.name
                label_data.to_crs(pyproj.CRS.from_epsg(4326))
                label_data.to_file(label_file)

        kwargs["label_vector_uri"] = label_file

    return make_ss_scene(**kwargs), kwargs, temp_label_file_manager


def compute_rastervision_evaluation_metrics(
    image_file: PATH_TYPE,
    prediction_file: PATH_TYPE,
    groundtruth_file: PATH_TYPE,
    class_names: list[str],
    vis_savefile: str = None,
):
    image_file = str(image_file)
    prediction_file = str(prediction_file)
    groundtruth_file = str(groundtruth_file)

    class_config = ClassConfig(names=class_names)
    class_config.ensure_null_class()

    logging.info("Creating prediction scenes")
    prediction_scene, prediction_kwargs, prediction_temp = make_ss_scene_vec_or_rast(
        class_config=class_config, image_file=image_file, label_file=prediction_file
    )
    gt_scene, gt_kwargs, gt_temp = make_ss_scene_vec_or_rast(
        class_config=class_config, image_file=image_file, label_file=groundtruth_file
    )
    # Do visualizations after creating the scene to ensure that the file has the 'class_id' field
    if vis_savefile is not None:
        logging.info("Visualizing")

        f, axs = plt.subplots(1, 2)
        plot_geodata(
            prediction_file if prediction_temp is None else prediction_temp.name,
            ax=axs[0],
        )
        plot_geodata(groundtruth_file if gt_temp is None else gt_temp.name, ax=axs[1])

        axs[0].set_title("Predictions")
        axs[1].set_title("Ground truth")
        plt.savefig(vis_savefile)
        plt.close()
        return

    logging.info("Getting label arrays")
    prediction_labels = prediction_scene.label_source.get_labels()
    gt_labels = gt_scene.label_source.get_labels()

    logging.info("Computing metrics")
    evaluator = SemanticSegmentationEvaluator(class_config)
    evaluation = evaluator.evaluate_predictions(
        ground_truth=gt_labels, predictions=prediction_labels
    )

    return evaluation


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
