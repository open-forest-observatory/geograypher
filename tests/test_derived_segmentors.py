import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import MultiPolygon, Polygon

from geograypher.predictors.derived_segmentors import RegionDetectionSegmentor


def create_vector_data_with_polygons(path, polygons, multi_polygons=None, labels=None):

    geometry = [Polygon(poly) for poly in polygons]
    if multi_polygons is not None:
        geometry.append(MultiPolygon([Polygon(poly) for poly in multi_polygons]))

    # If labels are not given, create a filler version
    if labels is None:
        labels = [0] * len(geometry)

    gdf = gpd.GeoDataFrame(
        {
            "geometry": geometry,
            "unique_ID": [f"{i:05}" for i in range(len(geometry))],
            "labels": labels,
            "score": np.random.random(len(geometry)),
        }
    )
    gdf.to_file(path)


class TestRegionDetectionSegmentor:

    @pytest.mark.parametrize("flat", (True, False))
    @pytest.mark.parametrize("geo_extension", (".gpkg", ".geojson", ".shp"))
    @pytest.mark.parametrize("im_extension", (".jpg", ".JPG", ".png", ".tif"))
    def test_detection_centers(self, tmp_path, flat, geo_extension, im_extension):

        polygons = [
            [(0, 50), (0, 60), (10, 60), (10, 50), (0, 50)],
            [(20, 20), (20, 30), (30, 30), (30, 20), (20, 20)],
            [(100, 50), (100, 60), (110, 60), (110, 50), (100, 50)],
        ]

        # Create a series of polygon files with content and empty images
        im_paths = []

        # Test that this works on a flat set of files, or on a nested set
        # of directories and files
        if flat:
            base_folder = tmp_path
            lookup_folder = tmp_path
            im_nested = tmp_path
            geo_nested = tmp_path
        else:
            base_folder = tmp_path / "images"
            lookup_folder = tmp_path / "geospatial"
            im_nested = base_folder / "mission" / "00"
            geo_nested = lookup_folder / "mission" / "00"
            geo_nested.mkdir(parents=True)

        for i in range(5):
            im_paths.append(im_nested / f"test_{i}{im_extension}")
            create_vector_data_with_polygons(
                geo_nested / f"test_{i}{geo_extension}", polygons
            )

        segmentor = RegionDetectionSegmentor(
            base_folder=base_folder,
            lookup_folder=lookup_folder,
            label_key=None,
            class_map=None,
            geo_file_extension=geo_extension,
        )

        for im_path in im_paths:

            # We can get the centers for each polygon
            centers = segmentor.get_detection_centers(im_path)
            assert centers.shape == (3, 2)

            # Check that the extracted centroids are where we expect
            for i, poly in enumerate(polygons):
                expected_centroid = Polygon(poly).centroid
                np.testing.assert_allclose(
                    centers[i], [expected_centroid.y, expected_centroid.x]
                )

    def test_empty(self, tmp_path):
        segmentor = RegionDetectionSegmentor(
            base_folder=tmp_path,
            lookup_folder=tmp_path,
            label_key=None,
            class_map=None,
            geo_file_extension=".gpkg",
        )
        centers = segmentor.get_detection_centers(str(tmp_path / "nonexistent.JPG"))
        assert centers.shape == (0, 2)

    @pytest.mark.parametrize("imshape", [(40, 40), (60, 40), (100, 120)])
    @pytest.mark.parametrize("geo_extension", (".gpkg", ".geojson", ".shp"))
    def test_segment_image(self, tmp_path, geo_extension, imshape):

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
        create_vector_data_with_polygons(
            tmp_path / f"test{geo_extension}",
            polygons,
            multi_polygons,
            labels=["APPLE", "APPLE", "PEAR", "FIG"],
        )

        segmentor = RegionDetectionSegmentor(
            base_folder=tmp_path,
            lookup_folder=tmp_path,
            label_key="labels",
            class_map={"FIG": 0, "PEAR": 1, "APPLE": 2},
            geo_file_extension=geo_extension,
        )

        mask = segmentor.segment_image(
            image=None,
            im_path=tmp_path / f"test.JPG",
            image_shape=imshape,
        )
        # Check the shape is (H, W, N indices) where each label class has its own
        # one-hot slice
        assert mask.shape == imshape + (3,)
        assert mask.dtype == bool

        # Check that the areas are correct. Because segment_image returns a
        # one_hot array, overlapping masks (like #3 overlapping #2) don't
        # affect the mask area
        assert (mask[..., 0]).sum() == 187
        assert (mask[..., 1]).sum() == 121
        assert (mask[..., 2]).sum() == 121 + 121

        # Check that the location is correct
        assert np.allclose(np.average(np.where(mask[..., 0]), axis=1), [5, 28])
        assert np.allclose(np.average(np.where(mask[..., 1]), axis=1), [30, 25])
        assert np.allclose(np.average(np.where(mask[..., 2]), axis=1), [15, 15])

    @pytest.mark.parametrize(
        "label_key,class_map,expected_str",
        (
            ["nonexistent", None, "not found in GDF columns"],
            ["labels", {"BETA": 0}, "keys in a GDF which were not in the class map"],
            ["labels", {"ALPHA": 1.5}, "not integer indices"],
            ["labels", {"ALPHA": "BETA"}, "not integer indices"],
        ),
    )
    def test_segment_image_errors(self, tmp_path, label_key, class_map, expected_str):
        """Test a few scenarios that we expect to error out."""

        create_vector_data_with_polygons(
            tmp_path / "test.gpkg",
            [[(0, 0), (0, 10), (10, 10), (10, 0), (0, 0)]],
            labels=["ALPHA"],
        )
        segmentor = RegionDetectionSegmentor(
            base_folder=tmp_path,
            lookup_folder=tmp_path,
            label_key=label_key,
            class_map=class_map,
        )
        with pytest.raises(ValueError) as ve:
            segmentor.segment_image(
                image=None,
                im_path=tmp_path / f"test.JPG",
                image_shape=(100, 100),
            )
        assert expected_str in str(ve.value)
