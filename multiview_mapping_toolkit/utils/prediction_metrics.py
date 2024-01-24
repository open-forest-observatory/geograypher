import logging
import tempfile
import typing
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pyproj
import rasterio as rio
from rasterstats import zonal_stats
from rasterio.plot import reshape_as_image
from rastervision.core.data import ClassConfig
from rastervision.core.data.utils import make_ss_scene
from rastervision.core.evaluation import SemanticSegmentationEvaluator
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from IPython.core.debugger import set_trace

from multiview_mapping_toolkit.config import (
    PATH_TYPE,
    TEN_CLASS_VIS_KWARGS,
    CLASS_ID_KEY,
    CLASS_NAMES_KEY,
)
from multiview_mapping_toolkit.utils.geospatial import (
    ensure_geometric_CRS,
    reproject_raster,
    get_overlap_vector,
    get_overlap_raster,
    coerce_to_geoframe,
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
        cb = ax.imshow(img, vmin=vmin, vmax=vmax, cmap=cmap)
        if single_channel:
            plt.colorbar(cb)
    else:
        data = gpd.read_file(filename)
        data.plot(class_column, ax=ax, vmin=vmin, vmax=vmax, cmap=cmap)


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
            if CLASS_ID_KEY not in label_data.columns:
                if len(label_data) != len(class_config.names) - 1:
                    # TODO see if "class" is there and create labels based on the class names
                    raise ValueError(
                        f"{CLASS_ID_KEY} not set and the number of rows is not the same as the number of classes"
                    )
                label_data[CLASS_ID_KEY] = label_data.index
                # Create a temp file where we add the class_ID field
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
    # Do visualizations after creating the scene to ensure that the file has the class_ID field
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


def compute_evaluation_metrics_no_rastervision(
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

            per_class_area, classes = cf_from_vector_vector(
                predicted_df=prediction_gdf,
                true_df=groundtruth_gdf,
                column_name=column_name,
            )
            # Compute the sum of labeled predictions per ground truth class
            labeled_areas = np.sum(per_class_area, axis=1)
            # Compute the area of each ground truth class
            total_areas = groundtruth_gdf.area.values
            # TODO this isn't correct if predictions are overlapping
            unlabeled_areas = total_areas - labeled_areas

            # Create a cf matrix with an un-labeled class
            cf_matrix = np.zeros((len(class_names) + 1, len(class_names) + 1))
            cf_matrix[:-1, :-1] = per_class_area
            cf_matrix[:-1, -1] = unlabeled_areas

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

    accuracy = np.sum(cf_matrix * np.eye(cf_matrix.shape[0]))

    print(f"Accuracy {accuracy}")

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
