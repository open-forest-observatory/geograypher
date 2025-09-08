from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import pyvista as pv

from geograypher.cameras.cameras import PhotogrammetryCamera, PhotogrammetryCameraSet


class MockDetector:
    def get_detection_centers(self, _):
        """Return some fixed detection centers for testing"""
        return np.array([[0, 0], [100, 100], [300, 300], [400, 400]])


def make_sample_camera(idx):
    """Create a simple camera looking down the z-axis, located at z=10"""
    cam_to_world = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 10], [0, 0, 0, 1]])
    return PhotogrammetryCamera(
        image_filename=f"/tmp/test_{idx}.jpg",
        cam_to_world_transform=cam_to_world,
        f=1000,
        cx=0,
        cy=0,
        image_width=800,
        image_height=800,
    )


@pytest.fixture
def sample_camera_set():
    # Test a set of multiple cameras, albeit copies of each other
    return PhotogrammetryCameraSet(cameras=[make_sample_camera(i) for i in range(5)])


@pytest.fixture
def sample_boundaries():
    """
    Create two simple planar surfaces for boundaries. Upper boundary
    at z=5, lower at z=0.
    """
    # z=0 plane
    plane0 = np.array(
        [
            [-10, -10, 0],
            [10, -10, 0],
            [10, 10, 0],
            [-10, 10, 0],
        ]
    )
    # Move up to z=5
    plane1 = plane0.copy()
    plane1[:, 2] = 5
    return (
        pv.PolyData(plane0).delaunay_2d(),
        pv.PolyData(plane1).delaunay_2d(),
    )


class TestCalcLineSegments:

    @pytest.mark.parametrize("to_file", [True, False])
    @pytest.mark.parametrize("ray_length_local", [100, 200, 300])
    def test_basic_line_segments(
        self, tmp_path, sample_camera_set, ray_length_local, to_file
    ):
        """Test basic line segment generation without boundaries"""

        detector = MockDetector()
        output = sample_camera_set.calc_line_segments(
            detector=detector,
            ray_length_local=ray_length_local,
            out_dir=tmp_path if to_file else None,
        )

        # Load the results if necessary
        if to_file:
            assert isinstance(output, Path)
            assert output.suffix == ".npz"
            assert output.is_file()
            result = np.load(output)
        else:
            result = output

        n_cameras = len(sample_camera_set.cameras)
        n_detections = len(detector.get_detection_centers(None))

        # Check shapes and basic properties
        assert "ray_starts" in result
        assert "ray_ends" in result
        assert "ray_IDs" in result
        N = n_cameras * n_detections
        assert result["ray_starts"].shape == (N, 3)
        assert result["ray_ends"].shape == (N, 3)
        assert result["ray_IDs"].shape == (N,)

        # Check that rays start at camera position
        assert np.allclose(result["ray_starts"][0], [0, 0, 10])

        # Check the endpoint that should be pointed straight down is
        # pointing straight down
        for i in range(3, N, n_detections):
            assert np.allclose(result["ray_ends"][i], [0, 0, 10 - ray_length_local])
        # Don't check specifics, but check that the maximumally tilted ray
        # has an angle over some threshold
        for i in range(0, N, n_detections):
            xyz = result["ray_ends"][i]
            hypotenuse = np.linalg.norm(xyz - np.array([0, 0, 10]))
            x_angle = np.arcsin(xyz[0] / hypotenuse)
            y_angle = np.arcsin(xyz[1] / hypotenuse)
            assert x_angle < -np.deg2rad(15)
            assert y_angle > np.deg2rad(15)

        # Check that ray lengths are scaled as expected
        ray_dirs = result["ray_ends"] - result["ray_starts"]
        assert np.allclose(np.linalg.norm(ray_dirs, axis=1), ray_length_local)

        # Check that ray_IDs are correct
        expected_IDs = sum(
            [[i] * n_detections for i in range(n_cameras)],
            start=[],
        )
        assert np.allclose(result["ray_IDs"], expected_IDs)

    def test_line_segments_with_boundaries(self, sample_camera_set, sample_boundaries):
        """Test line segment generation with boundary clipping"""
        detector = MockDetector()
        result = sample_camera_set.calc_line_segments(
            detector=detector, boundaries=sample_boundaries, ray_length_local=100
        )

        n_cameras = len(sample_camera_set.cameras)
        n_detections = len(detector.get_detection_centers(None))
        N = n_cameras * n_detections
        assert len(result["ray_starts"]) == N

        # Check that rays are clipped between boundaries
        z_coords = result["ray_ends"][:, 2]
        assert np.all(z_coords >= 0)  # Lower boundary
        assert np.all(z_coords <= 5)  # Upper boundary

    def test_angle_filtering(self, sample_camera_set):
        """
        Test filtering rays by angle from vertical. Based on the hardcoded
        camera positions, this should filter out the first two rays of each camera.
        """
        result = sample_camera_set.calc_line_segments(
            detector=MockDetector(),
            limit_angle_from_vert=np.deg2rad(15),
        )

        # Based on our angle limit and the hardcoded camera positions,
        # we expect to filter out two of the four detections per camera.
        n_cameras = len(sample_camera_set.cameras)
        n_detections = 2
        N = n_cameras * n_detections
        assert result["ray_starts"].shape == (N, 3)
        assert result["ray_ends"].shape == (N, 3)
        assert result["ray_IDs"].shape == (N,)

        # Check that ray_IDs are correct
        expected_IDs = sum(
            [[i] * n_detections for i in range(n_cameras)],
            start=[],
        )
        assert np.allclose(result["ray_IDs"], expected_IDs)

        # Calculate angles from vertical for resulting rays
        ray_vectors = result["ray_ends"] - result["ray_starts"]
        ray_dirs = ray_vectors / np.linalg.norm(ray_vectors, axis=1)[:, None]
        angles = np.arccos(np.abs(ray_dirs[:, 2]))

        # Check that all rays are within limit
        assert np.all(angles <= np.deg2rad(15))

    def test_empty_detections(self, sample_camera_set):
        """Test handling of empty detections"""
        detector = MagicMock()
        detector.get_detection_centers.return_value = np.array([])

        result = sample_camera_set.calc_line_segments(detector=detector)

        assert result["ray_starts"].shape == (0, 3)
        assert result["ray_ends"].shape == (0, 3)
        assert result["ray_IDs"].shape == (0,)


class TestTriangulateDetections:

    @pytest.mark.parametrize("ray_length", [100, 1000])
    @pytest.mark.parametrize("similarity_threshold", [0.1, 1.0])
    @pytest.mark.parametrize("limit_ray_length", [2, 100, None])
    @pytest.mark.parametrize("limit_angle", [np.deg2rad(20), None])
    @pytest.mark.parametrize("use_boundaries", [True, False])
    @pytest.mark.parametrize("transform", [lambda x: x**2, None])
    @pytest.mark.parametrize("louvain_resolution", [1, 10])
    @pytest.mark.parametrize("to_file", [True, False])
    def test_runs(
        self,
        sample_camera_set,
        sample_boundaries,
        ray_length,
        similarity_threshold,
        limit_ray_length,
        limit_angle,
        use_boundaries,
        transform,
        louvain_resolution,
        to_file,
        tmp_path,
    ):
        """
        Since triangulate_detections is an amalgam of tested components, just check
        that it runs on a variety of inputs and produces something of the right type.
        """

        points = sample_camera_set.triangulate_detections(
            detector=MockDetector(),
            ray_length_meters=ray_length,
            boundaries=sample_boundaries if use_boundaries else None,
            limit_ray_length_meters=limit_ray_length,
            limit_angle_from_vert=limit_angle,
            similarity_threshold_meters=similarity_threshold,
            transform=transform,
            louvain_resolution=louvain_resolution,
            out_dir=tmp_path if to_file else None,
        )

        # Check that we get some sort of (N, 3) array out
        assert isinstance(points, np.ndarray)
        assert points.shape[1] == 3

        if to_file:
            assert (tmp_path / "line_segments.npz").is_file()
            assert (tmp_path / "edge_weights.json").is_file()
            assert (tmp_path / "communities.npz").is_file()
