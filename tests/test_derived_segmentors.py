import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import MultiPolygon, Polygon

from geograypher.predictors.derived_segmentors import RegionDetectionSegmentor


def create_vector_data_with_polygons(path, polygons, multi_polygons=None):

    geometry = [Polygon(poly) for poly in polygons]
    if multi_polygons is not None:
        geometry.append(MultiPolygon([Polygon(poly) for poly in multi_polygons]))

    gdf = gpd.GeoDataFrame(
        {
            "geometry": geometry,
            "unique_ID": [f"{i:05}" for i in range(len(geometry))],
            "labels": [0] * len(geometry),
            "score": np.random.random(len(geometry)),
        }
    )
    driver_options = {
        ".gpkg": "GPKG",
        ".geojson": "GeoJSON",
        ".shp": "ESRI Shapefile",
    }
    gdf.to_file(path, driver=driver_options[path.suffix])


class TestRegionDetectionSegmentor:

    @pytest.mark.parametrize("multiple", (True, False))
    @pytest.mark.parametrize("geo_extension", (".gpkg", ".geojson", ".shp"))
    @pytest.mark.parametrize("im_extension", (".jpg", ".JPG", ".png", ".tif"))
    def test_segmentor(self, tmp_path, geo_extension, im_extension, multiple):

        polygons = [
            [(0, 0), (0, 10), (10, 10), (10, 0), (0, 0)],
            [(20, 20), (20, 30), (30, 30), (30, 20), (20, 20)],
            [(50, 50), (50, 60), (60, 60), (60, 50), (50, 50)],
        ]
        if multiple:
            for i in range(5):
                geo_path = tmp_path / f"test_{i}{geo_extension}"
                create_vector_data_with_polygons(geo_path, polygons)
            data_location = tmp_path
            image_names = [f"test_{i}{im_extension}" for i in range(5)]
        else:
            geo_path = tmp_path / f"test{geo_extension}"
            create_vector_data_with_polygons(geo_path, polygons)
            data_location = geo_path
            image_names = [f"test{im_extension}"]

        segmentor = RegionDetectionSegmentor(
            detection_file_or_folder=data_location,
            geo_file_extension=geo_extension,
            image_file_extension=im_extension,
        )

        for imname in image_names:
            # Each image is recorded
            assert imname in segmentor.image_names

            # Each image gets a len=3 group of polygons
            gdf = segmentor.grouped_labels_gdf.get_group(imname)
            assert len(gdf) == 3

            # We can get the centers for each polygon
            centers = segmentor.get_detection_centers(imname)
            assert centers.shape == (3, 2)

            # Check that the extracted centroids are where we expect
            for i, poly in enumerate(polygons):
                expected_centroid = Polygon(poly).centroid
                np.testing.assert_allclose(
                    centers[i], [expected_centroid.y, expected_centroid.x]
                )

    def test_empty(self, tmp_path):
        segmentor = RegionDetectionSegmentor(
            detection_file_or_folder=tmp_path,
            image_file_extension=".JPG",
        )
        assert segmentor.image_names == []
        centers = segmentor.get_detection_centers("nonexistent.JPG")
        assert centers.shape == (0, 2)

    @pytest.mark.parametrize("imshape", [(40, 40), (60, 40), (100, 120)])
    @pytest.mark.parametrize("geo_extension", (".gpkg", ".geojson", ".shp"))
    @pytest.mark.parametrize("im_extension", (".jpg", ".JPG", ".png", ".tif"))
    def test_segment_image(self, tmp_path, geo_extension, im_extension, imshape):

        # Create polygons, the third of which is overlapping the second
        polygons = [
            [(0, 0), (0, 10), (10, 10), (10, 0), (0, 0)],
            [(20, 20), (20, 30), (30, 30), (30, 20), (20, 20)],
            [(25, 20), (25, 30), (35, 30), (35, 20), (25, 20)],
        ]
        multi_polygons = [
            [(0, 20), (0, 30), (10, 30), (10, 20), (0, 20)],
            [(0, 26), (0, 36), (10, 36), (10, 26), (0, 26)],
        ]
        geo_path = tmp_path / f"test{geo_extension}"
        create_vector_data_with_polygons(geo_path, polygons, multi_polygons)

        segmentor = RegionDetectionSegmentor(
            detection_file_or_folder=geo_path,
            image_file_extension=im_extension,
        )

        mask = segmentor.segment_image(
            image=None,
            filename=f"test{im_extension}",
            image_shape=imshape,
        )
        # Check the shape is (H, W, N indices) where each mask has its own
        # one-hot slice
        assert mask.shape == imshape + (4,)
        assert mask.dtype == bool

        # Check that the areas are correct. Because segment_image returns a
        # one_hot array, overlapping masks (like #3 overlapping #2) don't
        # affect the mask area
        assert (mask[..., 0]).sum() == 121
        assert (mask[..., 1]).sum() == 121
        assert (mask[..., 2]).sum() == 121
        assert (mask[..., 3]).sum() == 187

        # Check that the location is correct
        assert np.allclose(np.average(np.where(mask[..., 0]), axis=1), [5, 5])
        assert np.allclose(np.average(np.where(mask[..., 1]), axis=1), [25, 25])
        assert np.allclose(np.average(np.where(mask[..., 2]), axis=1), [30, 25])
        assert np.allclose(np.average(np.where(mask[..., 3]), axis=1), [5, 28])
