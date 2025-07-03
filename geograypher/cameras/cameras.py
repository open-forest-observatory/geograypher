import hashlib
import json
import logging
import os
import re
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple, Union

import geopandas as gpd
import networkx
import numpy as np
import numpy.ma as ma
import pyvista as pv
from matplotlib import cm
from matplotlib.colors import Normalize
from pyvista import demos
from scipy.spatial.distance import pdist
from shapely import MultiPolygon, Point, Polygon, unary_union
from skimage.io import imread
from skimage.transform import resize
from tqdm import tqdm

from geograypher.constants import (
    DEFAULT_FRUSTUM_SCALE,
    EARTH_CENTERED_EARTH_FIXED_CRS,
    EXAMPLE_INTRINSICS,
    LAT_LON_CRS,
    PATH_TYPE,
)
from geograypher.predictors.derived_segmentors import TabularRectangleSegmentor
from geograypher.utils.files import ensure_containing_folder
from geograypher.utils.geometric import (
    angle_between,
    get_scale_from_transform,
    orthogonal_projection,
    projection_onto_plane,
)
from geograypher.utils.geospatial import convert_CRS_3D_points, ensure_projected_CRS
from geograypher.utils.image import get_GPS_exif
from geograypher.utils.numeric import (
    compute_approximate_ray_intersection,
    triangulate_rays_lstsq,
)
from geograypher.utils.visualization import safe_start_xvfb


class PhotogrammetryCamera:
    def __init__(
        self,
        image_filename: PATH_TYPE,
        cam_to_world_transform: np.ndarray,
        f: float,
        cx: float,
        cy: float,
        image_width: int,
        image_height: int,
        distortion_params: Dict[str, float] = {},
        lon_lat: Union[None, Tuple[float, float]] = None,
        local_to_epsg_4978_transform: Union[np.array, None] = None,
    ):
        """Represents the information about one camera location/image as determined by photogrammetry

        Args:
            image_filename (PATH_TYPE): The image used for reconstruction
            transform (np.ndarray): A 4x4 transform representing the camera-to-world transform
            f (float): Focal length in pixels
            cx (float): Principle point x (pixels) from center
            cy (float): Principle point y (pixels) from center
            image_width (int): Input image width pixels
            image_height (int): Input image height pixels
            distortion_params (dict, optional): Distortion parameters, currently unused
            lon_lat (Union[None, Tuple[float, float]], optional): Location, defaults to None
        """
        self.image_filename = image_filename
        self.cam_to_world_transform = cam_to_world_transform
        self.world_to_cam_transform = np.linalg.inv(cam_to_world_transform)
        self.f = f
        self.cx = cx
        self.cy = cy
        self.image_width = image_width
        self.image_height = image_height
        self.distortion_params = distortion_params
        self.local_to_epsg_4978_transform = local_to_epsg_4978_transform

        if lon_lat is None:
            self.lon_lat = (None, None)
        else:
            self.lon_lat = lon_lat

        self.image_size = (image_height, image_width)
        self.image = None
        self.cache_image = (
            False  # Only set to true if you can hold all images in memory
        )

    def get_camera_hash(self, include_image_hash: bool = False):
        """Generates a hash value for the camera's geometry and optionally includes the image

        Args:
            include_image_hash (bool, optional): Whether to include the image filename in the hash computation. Defaults to false.

        Returns:
            int: A hash value representing the current state of the camera
        """
        # Geometric information of hash
        transform_hash = self.cam_to_world_transform.tolist()
        camera_settings = {
            "transform": transform_hash,
            "f": self.f,
            "cx": self.cx,
            "cy": self.cy,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "distortion_params": self.distortion_params,
            "lon_lat": self.lon_lat,
        }

        # Include the image associated with the hash if specified
        if include_image_hash:
            camera_settings["image_filename"] = str(self.image_filename)

        camera_settings_data = json.dumps(camera_settings, sort_keys=True)
        hasher = hashlib.sha256()
        hasher.update(camera_settings_data.encode("utf-8"))

        return hasher.hexdigest()

    def get_camera_properties(self):
        """Returns the properties about a camera.

        Returns:
            dict: A dictionary containing the focal length, principal point coordinates,
                image height, image width, distortion parameters, and world_to_cam_transform.
        """
        camera_properties = {
            "focal_length": self.f,
            "principal_point_x": self.cx,
            "principal_point_y": self.cy,
            "image_height": self.image_height,
            "image_width": self.image_width,
            "distortion_params": self.distortion_params,
            "world_to_cam_transform": self.world_to_cam_transform,
        }
        return camera_properties

    def get_image(self, image_scale: float = 1.0) -> np.ndarray:
        # Check if the image is cached
        if self.image is None:
            image = imread(self.image_filename)
            if image.dtype == np.uint8:
                image = image / 255.0

            # Avoid unneccesary read if we have memory
            if self.cache_image:
                self.image = image
        else:
            image = self.image

        # Resizing is never cached, consider revisiting
        if image_scale != 1.0:
            image = resize(
                image,
                (int(image.shape[0] * image_scale), int(image.shape[1] * image_scale)),
            )

        return image

    def get_image_filename(self):
        return self.image_filename

    def get_image_size(self, image_scale=1.0):
        """Return image size, potentially scaled

        Args:
            image_scale (float, optional): How much to scale by. Defaults to 1.0.

        Returns:
            tuple[int]: (h, w) in pixels
        """
        # We should never have to deal with other cases if the reported size is accurate
        if self.image_size is not None:
            pass
        elif self.image is not None:
            self.image_size = self.image.shape[:2]
        else:
            image = self.get_image()
            self.image_size = image.shape[:2]

        return (
            int(self.image_size[0] * image_scale),
            int(self.image_size[1] * image_scale),
        )

    def get_lon_lat(self, negate_easting=True):
        """Return the lon, lat tuple, reading from exif metadata if neccessary"""
        if None in self.lon_lat:
            self.lon_lat = get_GPS_exif(self.image_filename)

            if negate_easting:
                self.lon_lat = (-self.lon_lat[0], self.lon_lat[1])

        return self.lon_lat

    def get_camera_location(self, get_z_coordinate: bool = False):
        """Returns a tuple of camera coordinates from the camera-to-world transfromation matrix.
        Args:
            get_z_coordinate (bool):
                Flag that user can set if they want z-coordinates. Defaults to False.
        Returns:
            Tuple[float, float (, float)]: tuple containing internal mesh coordinates of the camera
        """
        return (
            tuple(self.cam_to_world_transform[0:3, 3])
            if get_z_coordinate
            else tuple(self.cam_to_world_transform[0:2, 3])
        )

    def get_camera_view_angle(self, in_deg: bool = True) -> tuple:
        """Get the off-nadir pitch and yaw angles, computed geometrically from the photogrammtery result

        Args:
            in_deg (bool, optional): Return the angles in degrees rather than radians. Defaults to True.

        Returns:
            tuple: (pitch-from-nadir, yaw-from-nadir). Units are defined by in_deg parameter
        """
        # This is the origin, a point at one unit along the principal axis, a point one unit
        # up (-Y), and a point one unit right (+X)
        points_in_camera_frame = np.array(
            [[0, 0, 0, 1], [0, 0, 1, 1], [0, -1, 0, 1], [1, 0, 0, 1]]
        ).T

        # Transform the points first into the world frame and then into the earth-centered,
        # earth-fixed frame
        points_in_ECEF = (
            self.local_to_epsg_4978_transform
            @ self.cam_to_world_transform
            @ points_in_camera_frame
        )
        # Remove the homogenous coordinate and transpose
        points_in_ECEF = points_in_ECEF[:-1].T
        # Convert to shapely points
        points_in_ECEF = [Point(*point) for point in points_in_ECEF]
        # Convert to a dataframe
        points_in_ECEF = gpd.GeoDataFrame(
            geometry=points_in_ECEF, crs=EARTH_CENTERED_EARTH_FIXED_CRS
        )

        # Convert to lat lon
        points_in_lat_lon = points_in_ECEF.to_crs(LAT_LON_CRS)
        # Convert to a local projected CRS
        points_in_projected_CRS = ensure_projected_CRS(points_in_lat_lon)
        # Extract the geometry
        points_in_projected_CRS = np.array(
            [[p.x, p.y, p.z] for p in points_in_projected_CRS.geometry]
        )

        # Compute three vectors starting at the camera origin
        view_vector = points_in_projected_CRS[1] - points_in_projected_CRS[0]
        up_vector = points_in_projected_CRS[2] - points_in_projected_CRS[0]
        right_vector = points_in_projected_CRS[3] - points_in_projected_CRS[0]

        # The nadir vector points straight down
        NADIR_VEC = np.array([0, 0, -1])

        # For pitch, project the view vector onto the plane defined by the up vector and the nadir
        pitch_projection_view_vec = projection_onto_plane(
            view_vector, up_vector, NADIR_VEC
        )
        # For yaw, project the view vector onto the plane defined by the right vector and the nadir
        yaw_projection_view_vec = projection_onto_plane(
            view_vector, right_vector, NADIR_VEC
        )

        # Find the angle between these projected vectors and the nadir vector
        pitch_angle = angle_between(pitch_projection_view_vec, NADIR_VEC)
        yaw_angle = angle_between(yaw_projection_view_vec, NADIR_VEC)

        # Return in degrees if requested
        if in_deg:
            return (np.rad2deg(pitch_angle), np.rad2deg(yaw_angle))
        # Return in radians
        return (pitch_angle, yaw_angle)

    def check_projected_in_image(
        self, homogenous_image_coords: np.ndarray, image_size: Tuple[int, int]
    ):
        """Check if projected points are within the bound of the image and in front of camera

        Args:
            homogenous_image_coords (np.ndarray): The points after the application of K[R|t]. (3, n_points)
            image_size (Tuple[int, int]): The size of the image (width, height) in pixels

        Returns:
            np.ndarray: valid_points_bool, boolean array corresponding to which points were valid (n_points)
            np.ndarray: valid_image_space_points, float array of image-space coordinates for only valid points, (n_valid_points, 2)
        """
        img_width, image_height = image_size

        # Divide by the z coord to project onto the image plane
        image_space_points = homogenous_image_coords[:2] / homogenous_image_coords[2:3]
        # Transpose for convenience, (n_points, 3)
        image_space_points = image_space_points.T

        # We only want to consider points in front of the camera. Simple projection cannot tell
        # if a point is on the same ray behind the camera
        in_front_of_cam = homogenous_image_coords[2] > 0

        # Check that the point is projected within the image and is in front of the camera
        # Pytorch doesn't have a logical_and.reduce operator, so this is the equivilent using boolean multiplication
        valid_points_bool = (
            (image_space_points[:, 0] > 0)
            * (image_space_points[:, 1] > 0)
            * (image_space_points[:, 0] < img_width)
            * (image_space_points[:, 1] < image_height)
            * in_front_of_cam
        )

        # Extract the points that are valid
        valid_image_space_points = image_space_points[valid_points_bool, :].to(
            torch.int
        )
        # Return the boolean array
        valid_points_bool = valid_points_bool.cpu().numpy()
        valid_image_space_points = valid_image_space_points.cpu().numpy
        return valid_points_bool, valid_image_space_points

    def extract_colors(
        self, valid_bool: np.ndarray, valid_locs: np.ndarray, img: np.ndarray
    ):
        """_summary_

        Args:
            valid_bool (np.ndarray): (n_points,) boolean array cooresponding to valid points
            valid_locs (np.ndarray): (n_valid, 2) float array of image-space locations (x,y)
            img (np.ndarray): (h, w, n_channels) image to query from

        Returns:
            np.ma.array: (n_points, n_channels) One color per valid vertex. Points that were invalid are masked out
        """
        # Set up the data arrays
        colors_per_vertex = np.zeros((valid_bool.shape[0], img.shape[2]))
        mask = np.ones((valid_bool.shape[0], img.shape[2])).astype(bool)

        # Set the entries which are valid to false, meaning a valid entry in the masked array
        # TODO see if I can use valid_bool directly instead
        valid_inds = np.where(valid_bool)[0]
        mask[valid_inds, :] = False

        # Extract coordinates
        i_locs = valid_locs[:, 1]
        j_locs = valid_locs[:, 0]
        # Index based on the coordinates
        valid_color_samples = img[i_locs, j_locs, :]
        # Insert the valid samples into the array at the valid locations
        colors_per_vertex[valid_inds, :] = valid_color_samples
        # Convert to a masked array
        masked_color_per_vertex = ma.array(colors_per_vertex, mask=mask)
        return masked_color_per_vertex

    def project_mesh_verts(self, mesh_verts: np.ndarray, img: np.ndarray, device: str):
        """Get a color per vertex using only projective geometry, without considering occlusion or distortion

        Returns:
            np.ma.array: (n_points, n_channels) One color per valid vertex. Points that were invalid are masked out
        """
        # [R|t] matrix
        transform_3x4_world_to_cam = torch.Tensor(
            self.world_to_cam_transform[:3, :]
        ).to(device)
        K = torch.Tensor(
            [
                [self.f, 0, self.image_width / 2.0 + self.cx],
                [0, self.f, self.image_width + self.cy],
                [0, 0, 1],
            ],
            device=device,
        )
        # K[R|t], (3,4). Premultiplying these two matrices avoids doing two steps of projections with all points
        camera_matrix = K @ transform_3x4_world_to_cam

        # Add the extra dimension of ones for matrix multiplication
        homogenous_mesh_verts = torch.concatenate(
            (
                torch.Tensor(mesh_verts).to(device),
                torch.ones((mesh_verts.shape[0], 1)).to(device),
            ),
            axis=1,
        ).T

        # TODO review terminology
        homogenous_camera_points = camera_matrix @ homogenous_mesh_verts
        # Determine what points project onto the image and at what locations
        valid_bool, valid_locs = self.check_projected_in_image(
            projected_verts=homogenous_camera_points,
            image_size=(self.image_width, self.image_height),
        )
        # Extract corresponding colors from the image
        colors_per_vertex = self.extract_colors(valid_bool, valid_locs, img)

        return colors_per_vertex

    def get_pyvista_camera(self, focal_dist: float = 10) -> pv.Camera:
        """
        Get a pyvista camera at the location specified by photogrammetry.
        Note that there is no principle point and only the vertical field of view is set

        Args:
            focal_dist (float, optional): How far away from the camera the center point should be. Defaults to 10.

        Returns:
            pv.Camera: The pyvista camera from that viewpoint.
        """
        # Instantiate a new camera
        camera = pv.Camera()
        # Get the position as the translational part of the transform
        camera_position = self.cam_to_world_transform[:3, 3]
        # Get the look point by transforming a ray along the camera's Z axis into world
        # coordinates and then adding this to the location
        camera_look = camera_position + self.cam_to_world_transform[:3, :3] @ np.array(
            (0, 0, focal_dist)
        )
        # Get the up direction of the camera by finding which direction the -Y (image up) vector is transformed to
        camera_up = self.cam_to_world_transform[:3, :3] @ np.array((0, -1, 0))
        # Compute the vertical field of view
        vertical_FOV_angle = np.rad2deg(2 * np.arctan((self.image_height / 2) / self.f))

        # Set the values
        camera.focal_point = camera_look
        camera.position = camera_position
        camera.up = camera_up
        camera.view_angle = vertical_FOV_angle

        return camera

    def vis(self, plotter: pv.Plotter = None, frustum_scale: float = 0.1):
        """
        Visualize the camera as a frustum, at the appropriate translation and
        rotation and with the given focal length and aspect ratio.


        Args:
            plotter (pv.Plotter): The plotter to add the visualization to
            frustum_scale (float, optional): The length of the frustum in world units. Defaults to 0.5.
        """
        scaled_halfwidth = self.image_width / (self.f * 2)
        scaled_halfheight = self.image_height / (self.f * 2)

        scaled_cx = self.cx / self.f
        scaled_cy = self.cy / self.f

        right = scaled_cx + scaled_halfwidth
        left = scaled_cx - scaled_halfwidth
        top = scaled_cy + scaled_halfheight
        bottom = scaled_cy - scaled_halfheight

        vertices = (
            np.array(
                [
                    [0, 0, 0],
                    [
                        right,
                        top,
                        1,
                    ],
                    [
                        right,
                        bottom,
                        1,
                    ],
                    [
                        left,
                        bottom,
                        1,
                    ],
                    [
                        left,
                        top,
                        1,
                    ],
                ]
            ).T
            * frustum_scale
        )
        # Make the coordinates homogenous
        vertices = np.vstack((vertices, np.ones((1, 5))))

        # Project the vertices into the world cordinates
        projected_vertices = self.cam_to_world_transform @ vertices

        # Deal with the case where there is a scale transform
        if self.cam_to_world_transform[3, 3] != 1.0:
            projected_vertices /= self.cam_to_world_transform[3, 3]

        ## mesh faces
        faces = np.hstack(
            [
                [3, 0, 1, 2],  # side
                [3, 0, 2, 3],  # bottom
                [3, 0, 3, 4],  # side
                [3, 0, 4, 1],  # top
                [3, 1, 2, 3],  # endcap tiangle #1
                [3, 3, 4, 1],  # endcap tiangle #2
            ]
        )
        # All blue except the top (-Y) surface is red
        face_colors = np.array(
            [[0, 0, 1], [1, 0, 0], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]]
        ).astype(float)

        # Create a mesh for the camera frustum
        frustum = pv.PolyData(projected_vertices[:3].T, faces)
        # Unsure exactly what's going on here, but it's required for it to be valid
        frustum.triangulate()
        # Show the mesh with the given face colors
        # TODO understand how this understands it's face vs. vertex colors? Simply by checking the number of values?
        plotter.add_mesh(frustum, scalars=face_colors, rgb=True)

    def cast_rays(self, pixel_coords_ij: np.ndarray, line_length: float = 10):
        """Compute rays eminating from the camera

        Args:
            image_coords (np.ndarray): (n,2) array of (i,j) pixel coordinates in the image
            line_length (float, optional): How long the lines are. Defaults to 10. #TODO allow an array of different values

        Returns:
            np.array: The projected vertices, TODO
        """
        # Transform from i, j to x, y
        pixel_coords_xy = np.flip(pixel_coords_ij, axis=1)

        # Cast a ray from the center of the mesh for vis
        principal_point = np.array(
            [[self.image_width / 2.0 + self.cx, self.image_height / 2.0 + self.cy]]
        )
        centered_pixel_coords = pixel_coords_xy - principal_point
        scaled_pixel_coords = centered_pixel_coords / self.f

        n_points = len(scaled_pixel_coords)

        if n_points == 0:
            return

        line_verts = [
            np.array(
                [
                    [0, 0, 0, 1],
                    [
                        point[0] * line_length,
                        point[1] * line_length,
                        line_length,
                        1,
                    ],
                ]
            )
            for point in scaled_pixel_coords
        ]
        line_verts = np.concatenate(line_verts, axis=0).T

        projected_vertices = self.cam_to_world_transform @ line_verts

        # Handle scale in transform
        if self.cam_to_world_transform[3, 3] != 1.0:
            projected_vertices /= self.cam_to_world_transform[3, 3]

        projected_vertices = projected_vertices[:3, :].T

        return projected_vertices

    def vis_rays(
        self, pixel_coords_ij: np.ndarray, plotter: pv.Plotter, line_length: float = 10
    ):
        """Show rays eminating from the camera

        Args:
            image_coords (np.ndarray): (n,2) array of (i,j) pixel coordinates in the image
            plotter (pv.Plotter): Plotter to use.
            line_length (float, optional): How long the lines are. Defaults to 10. #TODO allow an array of different values
        """
        # If there are no detections, just skip it
        if len(pixel_coords_ij) == 0:
            return

        projected_vertices = self.cast_rays(
            pixel_coords_ij=pixel_coords_ij, line_length=line_length
        )
        n_points = int(projected_vertices.shape[0] / 2)

        lines = np.vstack(
            (
                np.full(n_points, fill_value=2),
                np.arange(0, 2 * n_points, 2),
                np.arange(0, 2 * n_points, 2) + 1,
            )
        ).T

        mesh = pv.PolyData(projected_vertices.copy(), lines=lines.copy())
        plotter.add_mesh(mesh)


class PhotogrammetryCameraSet:
    def __init__(
        self,
        cameras: Union[None, PhotogrammetryCamera, List[PhotogrammetryCamera]] = None,
        cam_to_world_transforms: Union[None, List[np.ndarray]] = None,
        intrinsic_params_per_sensor_type: Dict[int, Dict[str, float]] = {
            0: EXAMPLE_INTRINSICS
        },
        image_filenames: Union[List[PATH_TYPE], None] = None,
        lon_lats: Union[None, List[Union[None, Tuple[float, float]]]] = None,
        image_folder: PATH_TYPE = None,
        sensor_IDs: List[int] = None,
        validate_images: bool = False,
        local_to_epsg_4978_transform: Union[np.array, None] = None,
    ):
        """_summary_

        Args:
            cam_to_world_transforms (List[np.ndarray]): The list of 4x4 camera to world transforms
            intrinsic_params_per_sensor (Dict[int, Dict]): A dictionary mapping from an int camera ID to the intrinsic parameters
            image_filenames (List[PATH_TYPE]): The list of image filenames, ideally absolute paths
            lon_lats (Union[None, List[Union[None, Tuple[float, float]]]]): A list of lon,lat tuples, or list of Nones, or None
            image_folder (PATH_TYPE): The top level folder of the images
            sensor_IDs (List[int]): The list of sensor IDs, that index into the sensors_params_dict
            validate_images (bool, optional): Should the existance of the images be checked. Defaults to False.

        Raises:
            ValueError: _description_
        """
        # Create an object using the supplied cameras
        if cameras is not None:
            if isinstance(cameras, PhotogrammetryCamera):
                cameras = [cameras]
            self.cameras = cameras
            return

        # Standardization
        n_transforms = len(cam_to_world_transforms)

        # Create list of Nones for image filenames if not set
        if image_filenames is None:
            image_filenames = [None] * n_transforms

        if sensor_IDs is None and len(intrinsic_params_per_sensor_type) == 1:
            # Create a list of the only index if not set
            sensor_IDs = [
                list(intrinsic_params_per_sensor_type.keys())[0]
            ] * n_transforms
        elif len(sensor_IDs) != n_transforms:
            raise ValueError(
                f"Number of sensor_IDs ({len(sensor_IDs)}) is different than the number of transforms ({n_transforms})"
            )

        # If lon lats is None, set it to a list of Nones per transform
        if lon_lats is None:
            lon_lats = [None] * n_transforms

        if image_folder is None:
            # TODO set it to the least common ancestor of all filenames
            pass

        # Record the values
        # TODO see if we ever use these
        self.cam_to_world_transforms = cam_to_world_transforms
        self.intrinsic_params_per_sensor_type = intrinsic_params_per_sensor_type
        self.image_filenames = image_filenames
        self.lon_lats = lon_lats
        self.sensor_IDs = sensor_IDs
        self.image_folder = image_folder

        if validate_images:
            missing_images, invalid_images = self.find_mising_images()
            if len(missing_images) > 0:
                print(f"Deleting {len(missing_images)} missing images")
                valid_images = np.where(np.logical_not(invalid_images))[0]
                self.image_filenames = np.array(self.image_filenames)[
                    valid_images
                ].tolist()
                # Avoid calling .tolist() because this will recursively set all elements to lists
                # when this should be a list of np.arrays
                self.cam_to_world_transforms = [
                    x for x in np.array(self.cam_to_world_transforms)[valid_images]
                ]
                self.sensor_IDs = np.array(self.sensor_IDs)[valid_images].tolist()
                self.lon_lats = np.array(self.lon_lats)[valid_images].tolist()

        self.cameras = []

        for image_filename, cam_to_world_transform, sensor_ID, lon_lat in zip(
            self.image_filenames,
            self.cam_to_world_transforms,
            self.sensor_IDs,
            self.lon_lats,
        ):
            sensor_params = self.intrinsic_params_per_sensor_type[sensor_ID]
            # This means the sensor did not have enough parameters to be valid
            if sensor_params is None:
                continue

            new_camera = PhotogrammetryCamera(
                image_filename,
                cam_to_world_transform,
                lon_lat=lon_lat,
                local_to_epsg_4978_transform=local_to_epsg_4978_transform,
                **sensor_params,
            )
            self.cameras.append(new_camera)

    def __len__(self):
        return self.n_cameras()

    def __getitem__(self, slice):
        subset_cameras = self.cameras[slice]
        if isinstance(subset_cameras, PhotogrammetryCamera):
            # this is just one item indexed
            return subset_cameras
        # else, wrap the list of cameras in a CameraSet
        return PhotogrammetryCameraSet(subset_cameras)

    def get_image_folder(self):
        return self.image_folder

    def find_mising_images(self):
        invalid_mask = []
        for image_file in self.image_filenames:
            if not image_file.is_file():
                invalid_mask.append(True)
            else:
                invalid_mask.append(False)
        invalid_images = np.array(self.image_filenames)[np.array(invalid_mask)].tolist()

        return invalid_images, invalid_mask

    def n_cameras(self) -> int:
        """Return the number of cameras"""
        return len(self.cameras)

    def n_image_channels(self) -> int:
        """Return the number of channels in the image"""
        return 3

    def get_cameras_in_folder(self, folder: PATH_TYPE):
        """Return the camera set with cameras corresponding to images in that folder

        Args:
            folder (PATH_TYPE): The folder location

        Returns:
            PhotogrammetryCameraSet: A copy of the camera set with only the cameras from that folder
        """
        # Get the inds where that camera is in the folder
        imgs_in_folder_inds = [
            i
            for i in range(len(self.cameras))
            if self.cameras[i].image_filename.is_relative_to(folder)
        ]
        # Return the PhotogrammetryCameraSet with those subset of cameras
        subset_cameras = self.get_subset_cameras(imgs_in_folder_inds)
        return subset_cameras

    def get_cameras_matching_filename_regex(
        self, filename_regex: str
    ) -> "PhotogrammetryCameraSet":
        """Return the subset of cameras who's filenames match the provided regex

        Args:
            filename_regex (str): Regular expression passed to 're' engine

        Returns:
            PhotogrammetryCameraSet: Subset of cameras matching the regex
        """
        # Compute boolean array for which ones match
        imgs_matching_regex = [
            bool(re.search(filename_regex, str(filename)))
            for filename in self.image_filenames
        ]
        # Convert to integer indices within the set
        imgs_matching_regex_inds = np.where(imgs_matching_regex)[0]

        # Get the corresponding subset
        subset_cameras = self.get_subset_cameras(imgs_matching_regex_inds)
        return subset_cameras

    def get_subset_cameras(self, inds: List[int]):
        subset_camera_set = deepcopy(self)
        subset_camera_set.cameras = [subset_camera_set[i] for i in inds]
        return subset_camera_set

    def get_image_by_index(self, index: int, image_scale: float = 1.0) -> np.ndarray:
        return self[index].get_image(image_scale=image_scale)

    def get_camera_view_angles(self, in_deg: bool = True) -> List[Tuple[float]]:
        """Compute the pitch and yaw off-nadir for all cameras in the set

        Args:
            in_deg (bool, optional): Return the angles in degrees rather than radians. Defaults to True.

        Returns:
            List[Tuple[float]]: A list of (pitch, yaw) tuples for each camera.
        """
        return [
            camera.get_camera_view_angle(in_deg=in_deg)
            for camera in tqdm(self.cameras, desc="Computing view angles")
        ]

    def get_image_filename(self, index: Union[int, None], absolute=True):
        """Get the image filename(s) based on the index

        Args:
            index (Union[int, None]):
                Return the filename of the camera at this index, or all filenames if None.
                #TODO update to support lists of integer indices as well
            absolute (bool, optional):
                Return the absolute filepath, as oposed to the path relative to the image folder.
                Defaults to True.

        Returns:
            typing.Union[PATH_TYPE, list[PATH_TYPE]]:
                If an integer index is provided, one path will be returned. If None, a list of paths
                will be returned.
        """
        if index is None:
            return [
                self.get_image_filename(i, absolute=absolute)
                for i in range(len(self.cameras))
            ]

        filename = self.cameras[index].get_image_filename()
        if absolute:
            return Path(filename)
        else:
            # THIS IS A HACK UNTIL CONSULTING DAVID - it's necessary because the images I
            # ran segmentation on are not the same as those saved in camera.xml
            try:
                return Path(filename).relative_to(self.get_image_folder())
            except ValueError:
                path = self.get_image_folder() / filename.name
                assert path.is_file()
                return path

    def save_images(self, output_folder, copy=False, remove_folder: bool = True):
        if remove_folder:
            if os.path.isdir(output_folder):
                print(f"about to remove {output_folder}")
                shutil.rmtree(output_folder)

        for i in tqdm(
            range(len(self.cameras)),
            f"{'copying' if copy else 'linking'} images to {output_folder}",
        ):
            output_file = Path(
                output_folder, self.get_image_filename(i, absolute=False)
            )
            ensure_containing_folder(output_file)
            src_file = self.get_image_filename(i, absolute=True)
            if copy:
                try:
                    shutil.copy(src_file, output_file)
                except FileNotFoundError:
                    logging.warning(f"Could not find {src_file}")
            else:
                os.symlink(src_file, output_file)

    def get_lon_lat_coords(self):
        """Returns a list of GPS coords for each camera"""
        return [x.get_lon_lat() for x in self.cameras]

    def get_camera_locations(self, **kwargs):
        """
        Returns a list of camera locations for each camera.

        Args:
            **kwargs: Keyword arguments to be passed to the PhotogrammetryCamera.get_camera_location method.

        Returns:
            List[Tuple[float, float] or Tuple[float, float, float]]:
                List of tuples containing the camera locations.
        """
        return [x.get_camera_location(**kwargs) for x in self.cameras]

    def get_subset_ROI(
        self,
        ROI: Union[PATH_TYPE, gpd.GeoDataFrame, Polygon, MultiPolygon],
        buffer_radius: float = 0,
        is_geospatial: bool = None,
    ):
        """Return cameras that are within a radius of the provided geometry

        Args:
            geodata (Union[PATH_TYPE, gpd.GeoDataFrame, Polygon, MultiPolygon]):
                This can be a Geopandas dataframe, path to a geofile readable by geopandas, or
                Shapely Polygon/MultiPolygon information that can be loaded into a geodataframe
            buffer_radius (float, optional):
                Return points within this buffer of the geometry. Defaults to 0. Represents
                meters if ROI is geospatial.
            is_geospatial (bool, optional):
                A flag for user to indicate if ROI is geospatial or not; if no flag is provided,
                the flag is set if the provided geodata has a CRS.
        Returns:
            subset_camera_set (List[PhotogrammetryCamera]):
                List of cameras that fall within the provided ROI
        """
        # construct GeoDataFrame if not provided
        if isinstance(ROI, (Polygon, MultiPolygon)):
            # assume geodata is lat/lon if is_geospatial is True
            if is_geospatial:
                ROI = gpd.GeoDataFrame(crs=LAT_LON_CRS, geometry=[ROI])
            else:
                ROI = gpd.GeoDataFrame(geometry=[ROI])
        elif not isinstance(ROI, gpd.GeoDataFrame):
            # Read in the geofile
            ROI = gpd.read_file(ROI)

        if is_geospatial is None:
            is_geospatial = ROI.crs is not None

        if not is_geospatial:
            # get internal coordinate system camera locations
            image_locations = [Point(*x) for x in self.get_camera_locations()]
            image_locations_df = gpd.GeoDataFrame(geometry=image_locations)
        else:
            # Make sure it's a geometric (meters-based) CRS
            ROI = ensure_projected_CRS(ROI)
            # Read the locations of all the points
            image_locations = [Point(*x) for x in self.get_lon_lat_coords()]
            # Create a dataframe, assuming inputs are lat lon
            image_locations_df = gpd.GeoDataFrame(
                geometry=image_locations, crs=LAT_LON_CRS
            )
            image_locations_df.to_crs(ROI.crs, inplace=True)

        # Drop all the fields except the geometry for computational reasons
        ROI = ROI["geometry"]
        # Buffer out by the requested distance
        ROI = ROI.buffer(buffer_radius)
        # Merge the potentially-individual polygons to one
        # TODO Do experiments to see if this step should be before or after the buffer.
        # Counterintuitively, it seems that after is faster
        ROI = unary_union(ROI.tolist())

        # Determine the binary mask for which cameras are within the ROI
        cameras_in_ROI = image_locations_df.within(ROI).to_numpy()
        # Convert to the integer indices
        cameras_in_ROI = np.where(cameras_in_ROI)[0]
        # Get the corresponding subset of cameras
        subset_camera_set = self.get_subset_cameras(cameras_in_ROI)
        return subset_camera_set

    def triangulate_detections(
        self,
        detector: TabularRectangleSegmentor,
        boundaries: Union[None, tuple],
        transform_to_epsg_4978=None,
        similarity_threshold_meters: float = 0.1,
        louvain_resolution: float = 2,
        vis: bool = True,
        plotter: pv.Plotter = pv.Plotter(),
        vis_ray_length_meters: float = 200,
        vis_dir: PATH_TYPE = None,
    ) -> np.ndarray:
        """Take per-image detections and triangulate them to 3D locations

        Args:
            detector (TabularRectangleSegmentor):
                Produces detections per image using the get_detection_centers method
            boundaries (typing.Union[None, tuple]):
                If given, this should be [0] an upper surface and [1] a lower surface. These
                surfaces will be used to clip the triangulation rays
            transform_to_epsg_4978 (typing.Union[np.ndarray, None], optional):
                The 4x4 transform to earth centered earth fixed coordinates. Defaults to None.
            similarity_threshold_meters (float, optional):
                Consider rays a potential match if the distance between them is less than this
                value. Defaults to 0.1.
            louvain_resolution (float, optional):
                The resolution parameter of the networkx.louvain_communities function. Defaults to
                2.0.
            vis (bool, optional):
                Whether to show the detection projections and intersecting points. Defaults to True.
            plotter (pv.Plotter, optional):
                The plotter to add the visualizations to is vis=True. If not set, a new one will be
                created. Defaults to pv.Plotter().
            vis_ray_length_meters (float, optional):
                The length of the visualized rays in meters. Defaults to 200.
            vis_dir (PATH_TYPE, optional):
                Directory to save visualization geometries if not None. Defaults to None.

        Returns:
            np.ndarray:
                (n unique objects, 3), the 3D locations of the identified objects.
                If transform_to_epsg_4978 is set, then this is in (lat, lon, alt), if not, it's in the
                local coordinate system of the mesh
        """
        # Determine scale factor relating meters to internal coordinates
        meters_to_local_scale = 1 / get_scale_from_transform(transform_to_epsg_4978)
        similarity_threshold_local = similarity_threshold_meters * meters_to_local_scale
        vis_ray_length_local = vis_ray_length_meters * meters_to_local_scale

        # Record the lines corresponding to each detection and the associated image ID
        all_line_segments = []
        all_image_IDs = []

        # Iterate over the cameras
        for camera_ind in range(len(self)):
            # Get the image filename
            image_filename = str(self.get_image_filename(camera_ind, absolute=False))
            # Get the centers of associated detection from the detector
            # TODO, this only works with "detectors" that can look up the detections based on the
            # filename alone. In the future we might want to support real detectors that actually
            # use the image.
            detection_centers_pixels = detector.get_detection_centers(image_filename)
            # Get the individual camera
            camera = self[camera_ind]
            # Project rays given the locations of the detections in pixel coordinates
            line_segments = camera.cast_rays(
                pixel_coords_ij=detection_centers_pixels,
                line_length=vis_ray_length_local,
            )
            # If there are no detections, this will be None
            if line_segments is not None:
                # Record the line segments, which will be ordered as alternating (start, end) rows
                all_line_segments.append(line_segments)
                # Record which image ID generated each line
                all_image_IDs.append(
                    np.full(int(line_segments.shape[0] / 2), fill_value=camera_ind)
                )

        # Concatenate the lists of arrays into a single array
        all_line_segments = np.concatenate(all_line_segments, axis=0)
        all_image_IDs = np.concatenate(all_image_IDs, axis=0)

        # Get the starts and ends, which are alternating rows
        ray_starts = all_line_segments[0::2]
        segment_ends = all_line_segments[1::2]
        # Determine the direction
        ray_directions = segment_ends - ray_starts
        # Make the ray directions unit length
        ray_directions = ray_directions / np.linalg.norm(
            ray_directions, axis=1, keepdims=True
        )

        if boundaries is not None:
            upper = boundaries[0].multi_ray_trace(
                origins=ray_starts,
                directions=ray_directions,
                first_point=True,
                retry=True,
            )
            lower = boundaries[1].multi_ray_trace(
                origins=ray_starts,
                directions=ray_directions,
                first_point=True,
                retry=True,
            )
            # Find the ray indices that were found to intersect with both surfaces
            matched_set = list(set(upper[1]).intersection(set(lower[1])))

            # Downsample to only the relevant ray indices
            ray_directions = ray_directions[matched_set]

            # Iterate through the rays with intersections and update the starts and ends
            new_starts = []
            new_ends = []
            for ray_index in matched_set:
                new_starts.append(upper[0][np.where(upper[1] == ray_index)[0][0]])
                new_ends.append(lower[0][np.where(lower[1] == ray_index)[0][0]])
            ray_starts = np.array(new_starts)
            segment_ends = np.array(new_ends)

        assert ray_starts.shape == ray_directions.shape
        assert ray_starts.shape == segment_ends.shape

        # Compute the distance matrix of ray-ray intersections
        num_dets = ray_starts.shape[0]
        intersection_dists = np.full((num_dets, num_dets), fill_value=np.nan)

        # Calculate the upper triangular matrix of ray-ray interesections
        for i in tqdm(range(num_dets), desc="Calculating quality of ray intersections"):
            for j in range(i + 1, num_dets):
                # Extract starts and directions
                a0 = ray_starts[i]
                a1 = segment_ends[i]
                b0 = ray_starts[j]
                b1 = segment_ends[j]
                # TODO explore whether this could be vectorized
                _, _, dist = compute_approximate_ray_intersection(a0, a1, b0, b1)
                interesection_dists[i, j] = dist

        # Filter out intersections that are above the threshold distance
        intersection_dists[intersection_dists > similarity_threshold_local] = np.nan

        # Determine which intersections are valid, represented by finite values
        i_inds, j_inds = np.where(np.isfinite(intersection_dists))

        # Build a list of (i, j, info_dict) tuples encoding the valid edges and their intersection
        # distance
        positive_edges = [
            (i, j, {"weight": 1 / intersection_dists[i, j]})
            for i, j in zip(i_inds, j_inds)
        ]

        # Build a networkx graph. The nodes represent an individual detection while the edges
        # represent the quality of the matches between detections.
        graph = networkx.Graph(positive_edges)
        # Determine Louvain communities which are sets of nodes. Ideally, this represents a set of
        # detections that all correspond to one 3D object
        communities = networkx.community.louvain_communities(
            graph, weight="weight", resolution=louvain_resolution
        )
        # Sort the communities by size
        communities = sorted(communities, key=len, reverse=True)

        ## Triangulate the rays for each community to identify the 3D location
        community_points = []
        # Record the community IDs per detection
        community_IDs = np.full(num_dets, fill_value=np.nan)
        # Iterate over communities
        for community_ID, community in enumerate(communities):
            # Get the indices of the detections for that community
            community_detection_inds = np.array(list(community))
            # Record the community ID for the corresponding detection IDs
            community_IDs[community_detection_inds] = community_ID

            # Get the set of starts and directions for that community
            community_starts = ray_starts[community_detection_inds]
            community_directions = ray_directions[community_detection_inds]

            # Determine the least squares triangulation of the rays
            community_3D_point = triangulate_rays_lstsq(
                community_starts, community_directions
            )
            community_points.append(community_3D_point)

        # Stack all of the points into one vector
        community_points = np.vstack(community_points)

        # Show the rays and detections
        if vis:
            # Show the line segements
            # TODO: consider coloring these lines by community
            lines_mesh = pv.line_segments_from_points(all_line_segments)
            plotter.add_mesh(
                lines_mesh,
                scalars=community_IDs,
                label="Rays, colored by community ID",
            )
            # Show the triangulated communtities as red spheres
            detected_points = pv.PolyData(community_points)
            plotter.add_points(
                detected_points,
                color="r",
                render_points_as_spheres=True,
                point_size=10,
                label="Triangulated locations",
            )
            plotter.add_legend()

        # Save visualization geometries if vis_dir is given
        if vis_dir is not None:
            # Save line segments as cylinders colored by community_IDs
            cylinders = []
            norm = Normalize(
                vmin=np.nanmin(community_IDs), vmax=np.nanmax(community_IDs)
            )
            cmap = cm.get_cmap("tab20")
            for i, (start, end, comm_id) in enumerate(
                zip(ray_starts, segment_ends, community_IDs)
            ):
                # Create a cylinder between start and end
                center = (start + end) / 2
                direction = end - start
                height = np.linalg.norm(direction)
                if height == 0:
                    continue
                direction = direction / height
                cyl = pv.Cylinder(
                    center=center,
                    direction=direction,
                    radius=0.05,
                    height=height,
                    resolution=4,
                    capping=True,
                )
                color = (np.array(cmap(norm(comm_id)))[:3] * 255).astype(np.uint8)
                cyl["scalars"] = np.full(cyl.n_points, comm_id)
                cyl.point_data["RGB"] = np.tile(color, (cyl.n_points, 1))
                cylinders.append(cyl)
            if cylinders:
                cylinder_polydata = cylinders[0]
                for c in cylinders[1:]:
                    cylinder_polydata = cylinder_polydata.merge(c)
                print(f"Saving visualized cylinders to {vis_dir / 'rays.ply'}")
                cylinder_polydata.save(vis_dir / "rays.ply", texture="RGB")
            # Save community_points as red cubes
            cubes = []
            for comm_id, pt in enumerate(community_points):
                cube = pv.Cube(center=pt, x_length=0.2, y_length=0.2, z_length=0.2)
                color = (np.array(cmap(norm(comm_id)))[:3] * 255).astype(np.uint8)
                cube.point_data["RGB"] = np.tile(color, (cube.n_points, 1))
                cubes.append(cube)
            if cubes:
                cube_polydata = cubes[0]
                for s in cubes[1:]:
                    cube_polydata = cube_polydata.merge(s)
                print(f"Saving visualized cubes to {vis_dir / 'points.ply'}")
                cube_polydata.save(vis_dir / "points.ply", texture="RGB")

        # Convert the intersection points from the local mesh coordinate system to lat lon
        if transform_to_epsg_4978 is not None:
            # Append a column of all ones to make the homogenous coordinates
            community_points_homogenous = np.concatenate(
                [community_points, np.ones_like(community_points[:, 0:1])], axis=1
            )
            # Use the transform matrix to transform the points into the earth centered, earth fixed
            # frame, EPSG:4978
            community_points_epsg_4978 = (
                transform_to_epsg_4978 @ community_points_homogenous.T
            ).T
            # Convert the points from earth centered, earth fixed frame to lat lon
            community_points_lat_lon = convert_CRS_3D_points(
                community_points_epsg_4978,
                input_CRS=EARTH_CENTERED_EARTH_FIXED_CRS,
                output_CRS=LAT_LON_CRS,
            )
            # Set the community points to lat lon
            community_points = community_points_lat_lon

        # Return the 3D locations of the community points
        return community_points

    def vis(
        self,
        plotter: pv.Plotter = None,
        add_orientation_cube: bool = False,
        show: bool = False,
        frustum_scale: float = None,
        force_xvfb: bool = False,
        interactive_jupyter: bool = False,
    ):
        """Visualize all the cameras

        Args:
            plotter (pv.Plotter):
                Plotter to add the cameras to. If None, will be created and then plotted
            add_orientation_cube (bool, optional):
                Add a cube to visualize the coordinate system. Defaults to False.
            show (bool, optional):
                Show the results instead of waiting for other content to be added
            frustum_scale (float, optional):
                Size of cameras in world units. If None, will set to 1/120th of the maximum distance
                between two cameras.
            force_xvfb (bool, optional):
                Force a headless rendering backend
            interactive_jupyter (bool, optional):
                Will allow you to interact with the visualization in your notebook if supported by
                the notebook server. Otherwise will fail. Only applicable if `show=True`. Defaults
                to False.

        """

        if plotter is None:
            plotter = pv.Plotter()
            show = True

        # Determine pairwise distance between each camera and set frustum_scale to 1/120th of the maximum distance found
        if frustum_scale is None:
            if self.n_cameras() >= 2:
                camera_translation_matrices = np.array(
                    [transform[:3, 3] for transform in self.cam_to_world_transforms]
                )
                distances = pdist(camera_translation_matrices, metric="euclidean")
                max_distance = np.max(distances)
                frustum_scale = (
                    (max_distance / 120) if max_distance > 0 else DEFAULT_FRUSTUM_SCALE
                )
            # else, set it to a default
            else:
                frustum_scale = DEFAULT_FRUSTUM_SCALE

        for camera in self.cameras:
            camera.vis(plotter, frustum_scale=frustum_scale)
        if add_orientation_cube:
            # TODO Consider adding to a freestanding vis module
            ocube = demos.orientation_cube()
            plotter.add_mesh(ocube["cube"], show_edges=True)
            plotter.add_mesh(ocube["x_p"], color="blue")
            plotter.add_mesh(ocube["x_n"], color="blue")
            plotter.add_mesh(ocube["y_p"], color="green")
            plotter.add_mesh(ocube["y_n"], color="green")
            plotter.add_mesh(ocube["z_p"], color="red")
            plotter.add_mesh(ocube["z_n"], color="red")
            plotter.show_axes()

        if show:
            if force_xvfb:
                safe_start_xvfb()
            plotter.show(jupyter_backend="trame" if interactive_jupyter else "static")
