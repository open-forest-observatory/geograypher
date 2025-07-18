import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Polygon

from geograypher.predictors.derived_segmentors import RegionDetectionSegmentor


def create_gpkg_with_polygons(gpkg_path, polygons):
    gdf = gpd.GeoDataFrame(
        {
            "geometry": [Polygon(poly) for poly in polygons],
            "unique_ID": [f"{i:05}" for i in range(len(polygons))],
            "labels": [0] * len(polygons),
            "score": np.random.random(len(polygons)),
        }
    )
    gdf.to_file(gpkg_path, driver="GPKG")


class TestRegionDetectionSegmentor:

    @pytest.mark.parametrize("multiple", (True, False))
    @pytest.mark.parametrize("extension", (".jpg", ".JPG", ".png", ".tif"))
    def test_segmentor(self, tmp_path, extension, multiple):

        polygons = [
            [(0, 0), (0, 10), (10, 10), (10, 0), (0, 0)],
            [(20, 20), (20, 30), (30, 30), (30, 20), (20, 20)],
            [(50, 50), (50, 60), (60, 60), (60, 50), (50, 50)],
        ]
        if multiple:
            for i in range(5):
                gpkg_path = tmp_path / f"test_{i}.gpkg"
                create_gpkg_with_polygons(gpkg_path, polygons)
            gpkg_location = tmp_path
            image_names = [f"test_{i}{extension}" for i in range(5)]
        else:
            gpkg_path = tmp_path / f"test.gpkg"
            create_gpkg_with_polygons(gpkg_path, polygons)
            gpkg_location = gpkg_path
            image_names = [f"test{extension}"]

        segmentor = RegionDetectionSegmentor(
            detection_file_or_folder=gpkg_location,
            image_file_extension=extension,
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
    @pytest.mark.parametrize("extension", (".jpg", ".JPG", ".png", ".tif"))
    def test_segment_image_basic(self, tmp_path, extension, imshape):

        # Create polygons, the third of which is overlapping the second
        polygons = [
            [(0, 0), (0, 10), (10, 10), (10, 0), (0, 0)],
            [(20, 20), (20, 30), (30, 30), (30, 20), (20, 20)],
            [(25, 20), (25, 30), (35, 30), (35, 20), (25, 20)],
        ]
        gpkg_path = tmp_path / "test.gpkg"
        create_gpkg_with_polygons(gpkg_path, polygons)

        segmentor = RegionDetectionSegmentor(
            detection_file_or_folder=gpkg_path,
            image_file_extension=extension,
        )

        mask = segmentor.segment_image(
            image=None,
            filename=f"test{extension}",
            image_shape=imshape,
        )
        # Check shape
        assert mask.shape == imshape
        # Check that only 0, 1, and nan are present
        unique_labels = set(np.unique(mask[~np.isnan(mask)]))
        assert unique_labels == {0, 1, 2}

        # Check that the areas are correct
        assert (mask == 0).sum() == 121
        assert (mask == 1).sum() == 55
        assert (mask == 2).sum() == 121

        # Check that the location is correct
        assert np.allclose(np.average(np.where(mask == 0), axis=1), [5, 5])
        assert np.allclose(np.average(np.where(mask == 1), axis=1), [22, 25])
        assert np.allclose(np.average(np.where(mask == 2), axis=1), [30, 25])
