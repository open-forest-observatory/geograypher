import hashlib
import json
import logging
import os
import re
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import geopandas as gpd
import networkx
import numpy as np
import numpy.ma as ma
import pyproj
import pyvista as pv
from pyvista import demos
from scipy.spatial.distance import pdist
from shapely import MultiPolygon, Point, Polygon, unary_union
from skimage.io import imread
from skimage.transform import resize, warp
from tqdm import tqdm

from geograypher.constants import (
    DEFAULT_FRUSTUM_SCALE,
    EARTH_CENTERED_EARTH_FIXED_CRS,
    EXAMPLE_INTRINSICS,
    LAT_LON_CRS,
    PATH_TYPE,
)
from geograypher.predictors.derived_segmentors import (
    RegionDetectionSegmentor,
    TabularRectangleSegmentor,
)
from geograypher.utils.files import ensure_containing_folder
from geograypher.utils.geometric import (
    angle_between,
    clip_line_segments,
    get_scale_from_transform,
    orthogonal_projection,
    projection_onto_plane,
)
from geograypher.utils.geospatial import convert_CRS_3D_points, ensure_projected_CRS
from geograypher.utils.image import flexible_inputs_warp, get_GPS_exif
from geograypher.utils.indexing import inverse_map_interpolation
from geograypher.utils.numeric import (
    calc_communities,
    calc_graph_weights,
    compute_approximate_ray_intersections,
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
        self._local_to_epsg_4978_transform = local_to_epsg_4978_transform

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

    def get_camera_location(
        self, get_z_coordinate: bool = False, as_CRS: Optional[pyproj.CRS] = None
    ):
        """Returns a tuple of camera coordinates from the camera-to-world transformation matrix.
        Args:
            get_z_coordinate (bool):
                Flag that user can set if they want z-coordinates. Defaults to False.
            as_CRS (Optional[pyproj.CRS]):
                If given, return the points in the given CRS. If not given,
                return points in the default frame (Metashape local)
        Returns:
            Tuple[float, float (, float)]: tuple containing internal mesh coordinates of the camera
        """

        if as_CRS is None:
            point = self.cam_to_world_transform[0:3, 3]
        else:
            transformer = pyproj.Transformer.from_crs(
                EARTH_CENTERED_EARTH_FIXED_CRS, as_CRS
            )
            cam_in_ECEF = (
                self._local_to_epsg_4978_transform
                @ self.cam_to_world_transform
                @ np.array([[0, 0, 0, 1]]).T
            )
            point = transformer.transform(
                xx=cam_in_ECEF[0, 0],
                yy=cam_in_ECEF[1, 0],
                zz=cam_in_ECEF[2, 0],
            )
        return tuple(point) if get_z_coordinate else tuple(point[:2])

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
            self._local_to_epsg_4978_transform
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

    def get_local_to_epsg_4978_transform(self) -> np.ndarray:
        """
        Return the 4x4 homogenous transform mapping from the local coordinates used for
        photogrammetry to the earth-centered, earth-fixed coordinate reference system defined by
        EPSG:4978 (https://epsg.io/4978).

        Returns:
            np.ndarray:
                The transform in the form:
                   [R | t]
                   [0 | 1]
                When a homogenous vector is multiplied on the right of this matrix, it is
                transformed from the local coordinate frame to EPSG:4978. Conversely, the inverse
                of this matrix can be used to map from EPSG:4879 to local coordinates.
        """
        return self._local_to_epsg_4978_transform

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

    def get_vis_mesh(self, frustum_scale: float = 0.1) -> pv.PolyData:
        """Get this camera as a mesh representation.

        Args:
            frustum_scale (float, optional): Size of cameras in world units.

        Returns (PolyData): blue mesh of the camera as a frustum with a
            red face indicating the image top.
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
                    [right, top, 1],
                    [right, bottom, 1],
                    [left, bottom, 1],
                    [left, top, 1],
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
                [3, 1, 2, 3],  # endcap triangle #1
                [3, 3, 4, 1],  # endcap triangle #2
            ]
        )
        # All blue except the top (-Y) surface is red
        face_colors = np.array(
            [
                [0, 0, 255],
                [255, 0, 0],
                [0, 0, 255],
                [0, 0, 255],
                [0, 0, 255],
                [0, 0, 255],
            ]
        ).astype(np.uint8)

        # Create a mesh for the camera frustum
        frustum = pv.PolyData(projected_vertices[:3].T, faces)
        # Unsure exactly what's going on here, but it's required for it to be valid
        frustum.triangulate()

        # Assign the face colors to the mesh
        frustum["RGB"] = pv.pyvista_ndarray(face_colors)

        return frustum

    def vis(self, plotter: pv.Plotter = None, frustum_scale: float = 0.1):
        """
        Visualize the camera as a frustum, at the appropriate translation and
        rotation and with the given focal length and aspect ratio.

        Args:
            plotter (pv.Plotter): The plotter to add the visualization to
            frustum_scale (float, optional): The length of the frustum in world units.
        """

        mesh = self.get_vis_mesh(frustum_scale)

        # Show the mesh with the given face colors
        plotter.add_mesh(
            mesh,
            scalars="RGB",
            rgb=True,
        )

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

        def scale_vector(point, line_length):
            """
            Helper function to normalize the direction vector and scale it so
            the final vector magnitude is line_length.
            """
            # Expand the (x, y) homogenous coordinates to (x, y, 1) and normalize
            homogeneous = np.hstack([point, 1])
            norm = np.linalg.norm(homogeneous)
            return (homogeneous / norm) * line_length

        line_verts = [
            np.array(
                [
                    [0, 0, 0, 1],
                    np.hstack([scale_vector(point, line_length), 1]),
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
        cam_to_world_transforms: Optional[List[np.ndarray]] = None,
        intrinsic_params_per_sensor_type: Dict[int, Dict[str, float]] = {
            0: EXAMPLE_INTRINSICS
        },
        image_filenames: Optional[List[PATH_TYPE]] = None,
        lon_lats: Optional[List[Union[None, Tuple[float, float]]]] = None,
        image_folder: Optional[PATH_TYPE] = None,
        sensor_IDs: Optional[List[int]] = None,
        validate_images: bool = False,
        local_to_epsg_4978_transform: np.ndarray = np.eye(4),
    ):
        """Create a camera set, representing multiple cameras in a common global coordinate frame.

        Args:
            cam_to_world_transforms (List[np.ndarray]): The list of 4x4 camera to world transforms
            intrinsic_params_per_sensor (Dict[int, Dict]): A dictionary mapping from an int camera ID to the intrinsic parameters
            image_filenames (List[PATH_TYPE]): The list of image filenames, ideally absolute paths
            lon_lats (Union[None, List[Union[None, Tuple[float, float]]]]): A list of lon,lat tuples, or list of Nones, or None
            image_folder (PATH_TYPE): The top level folder of the images
            sensor_IDs (List[int]): The list of sensor IDs, that index into the sensors_params_dict
            validate_images (bool, optional): Should the existance of the images be checked.
                Any image_filenames that do not exist will be dropped, leaving a CameraSet only
                containing existing images. Defaults to False.
            local_to_epsg_4978_transform (np.ndarray):
                A 4x4 transform mapping coordinates from the local frame of the camera set into the
                global earth-centered, earth-fixed coordinate frame EPSG:4978.

        Raises:
            ValueError: If the number of sensor IDs is different than the number of transforms.
        """
        # Record the values
        self._local_to_epsg_4978_transform = local_to_epsg_4978_transform
        # Save parameters used for caching distortion products
        self._maps_ideal_to_warped = {}
        self._maps_warped_to_ideal = {}

        # Create an object using the supplied cameras
        if cameras is not None:
            if isinstance(cameras, PhotogrammetryCamera):
                self.image_folder = Path(cameras.image_filename).parent
                cameras = [cameras]
            else:
                self.image_folder = Path(
                    os.path.commonpath([str(cam.image_filename) for cam in cameras])
                )
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

        # Record values for the future
        self.cam_to_world_transforms = cam_to_world_transforms
        self.intrinsic_params_per_sensor_type = intrinsic_params_per_sensor_type
        self.image_filenames = image_filenames
        self.lon_lats = lon_lats
        self.sensor_IDs = sensor_IDs
        self.image_folder = image_folder

        if validate_images:
            missing_images, invalid_images = self.find_missing_images()
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
        return PhotogrammetryCameraSet(
            subset_cameras,
            local_to_epsg_4978_transform=self._local_to_epsg_4978_transform,
        )

    def get_image_folder(self):
        return self.image_folder

    def find_missing_images(self):
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
            return Path(filename).relative_to(self.get_image_folder())

    def get_local_to_epsg_4978_transform(self):
        """
        Return the 4x4 homogenous transform mapping from the local coordinates used for
        photogrammetry to the earth-centered, earth-fixed coordinate reference system defined by
        EPSG:4978 (https://epsg.io/4978).

        Returns:
            np.ndarray:
                The transform in the form:
                   [R | t]
                   [0 | 1]
                When a homogenous vector is multiplied on the right of this matrix, it is
                transformed from the local coordinate frame to EPSG:4978. Conversely, the inverse
                of this matrix can be used to map from EPSG:4879 to local coordinates.
        """
        return self._local_to_epsg_4978_transform

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
        return [cam.get_camera_location(**kwargs) for cam in self.cameras]

    def distortion_key(
        self, parameters: Dict[str, float], image_scale: float = 1.0
    ) -> str:
        """
        Make a repeatable string key out of a distortion parameter dict
        so that we can cache results by distortion parameters. We are
        deciding that precision in the distortion parameters after 8
        decimal points will be ignored in the caching process.

        Arguments:
            parameters (dict): Distortion parameters are assumed to be stored
                "name": float(value)
            image_scale (float, optional): If we want to warp a downsampled
                version of an image, we need to track that mapping seperately
                because downsampling affects the pixel to pixel mapping.
                Include the image scale (0-1 float) along with the distortion
                parameters to make a key.

        Returns:
            A repeatable string key based on the distortion values.
        """
        keys = sorted(parameters.keys())
        strings = [f"{key}:{parameters[key]:.8f}" for key in keys] + [
            f"image_scale:{image_scale:.8f}"
        ]
        return "|".join(strings)

    def make_distortion_map(
        self,
        camera: PhotogrammetryCamera,
        inversion_downsample: int,
        image_scale: float = 1.0,
    ) -> None:
        """
        Cache a map connecting locations in one image to locations in another.
        The basic construction is the pixel position of the map is the position
        in the destination image, and the value of the map is the position in
        the source image. Therefore if location [20, 30] has value [22.2, 28.4],
        it means that the destination image pixel [20, 30] will be sampled from
        the source image at pixel [22.2, 28.4] (the sampler can choose to snap
        to the closest integer value or interpolate nearby pixels).

        Arguments:
            camera (PhotogrammetryCamera): Camera with parameters that
                define the warp process. These include image size, principal
                point, focal length, and distortion parameters.
            inversion_downsample (int): Downsample the inverse map process in
                order to reduce computation, at high res the resulting map
                trafeoffs should be minimal.
            image_scale: (float, optional) 0-1 fraction of the original image
                size, used when warping/dewarping downsampled images. See
                pix2face render_img_scale for an example.

        Caches:
            In self._maps_ideal_to_warped, stores a map of the structure
                discussed above, keyed by self.distortion_key(params) so it can
                be accessed for cameras using the same params.
            In self._maps_warped_to_ideal, stores an inverted version
        """

        # Sample over the ideal pixels, shape (H, W)
        im_h, im_w = camera.image_size
        if np.isclose(image_scale, 1.0):
            h_range = np.arange(im_h)
            w_range = np.arange(im_w)
        else:
            # In order to get a distortion map for a downsampled image, you
            # need to run the ideal_to_warped equation over the same original
            # range (0...im_h) because the radius of that range relates
            # directly to the warping. However, do it in fewer steps (step
            # size > 1) to reflect the downsampling.
            kwargs = {"start": 1 / (2 * image_scale), "step": 1 / image_scale}
            h_range = np.arange(stop=im_h, **kwargs)[: int(im_h * image_scale)]
            w_range = np.arange(stop=im_w, **kwargs)[: int(im_w * image_scale)]
        rows, cols = np.meshgrid(h_range, w_range, indexing="ij")

        # Fill the (H, W) elements with the (i, j) distorted values at those locations
        warp_cols, warp_rows = self.ideal_to_warped(camera, cols, rows)

        # If dealing with downsampled images, now that we ran the warp equation,
        # rescale the results to be within the new scale. If the original im_h
        # is 1000 and image_scale is 0.5, we need to run ideal_to_warped on
        # 0-1000 (for radius reasons) and then scale those results to 0-500.
        if not np.isclose(image_scale, 1.0):
            warp_cols *= image_scale
            warp_rows *= image_scale

        # Cache this mapping as (2, H, W)
        dkey = self.distortion_key(camera.distortion_params, image_scale)
        self._maps_ideal_to_warped[dkey] = np.stack([warp_rows, warp_cols], axis=0)
        # Invert the warp map and cache as (2, H, W)
        self._maps_warped_to_ideal[dkey] = inverse_map_interpolation(
            self._maps_ideal_to_warped[dkey],
            downsample=inversion_downsample,
        )

    def ideal_to_warped(
        self, camera: PhotogrammetryCamera, xpix: np.ndarray, ypix: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply the given camera's distortion parameters to map pixels in an
        ideal pinhole camera to pixel locations in the warped/distorted
        image.

        NOTE: Some standards apparently can define their distortion models
        from warped to ideal, if we end up handling cameras like that we can
        make the use-case more flexible.

        Arguments:
            camera (PhotogrammetryCamera): Camera with parameters that
                define the warp process. These include image size, principal
                point, focal length, and distortion parameters.
            xpix (numpy array): Array of unknown shape, must match ypix
            ypix (numpy array): Array of unknown shape, must match xpix

        Returns:
            Tuple of numpy arrays of warped x pixels and y pixels
            [0] warped xpix
            [1] warped ypix
        """
        raise NotImplementedError(
            f"ideal_to_warped not implemented for {self.__class__}."
        )

    def warp_dewarp_image(
        self,
        camera: PhotogrammetryCamera,
        input_image: np.ndarray,
        fill_value: float = 0.0,
        inversion_downsample: int = 8,
        interpolation_order: int = 1,
        warped_to_ideal: bool = True,
        image_scale: float = 1.0,
    ) -> np.ndarray:
        """
        Either apply a camera's distortion model to go from ideal→warped or
        undo the camera's distortion model to go from warped→ideal, depending
        on the warped_to_ideal flag.

        Pixels in the output image that do not correspond to any input pixel
        are set to the fill value.

        Note that the warp map is cached, so the first call will take longer
        and subsequent calls should be much faster.

        Arguments:
            camera (PhotogrammetryCamera): Camera with parameters that
                define the warp process. These include image size, principal
                point, focal length, and distortion parameters.
            input_image (np.ndarray): (I, J, 3) Input image
            fill_value (int, optional): Value to use for pixels in the
                output image that are not mapped from the input. Defaults to 0.
            inversion_downsample (int, optional): The distortion map creation
                process is too heavyweight for really high-res images,
                downsampling the inversion process gets a similar result with
                less computation.
            interpolation_order (int, optional):
                The order of the interpolation. 0 is nearest neighbor and should be used for discrete
                textures like pix2face masks. 1 can be used for data representing continious
                quantities. Defaults to 1.
            warped_to_ideal (bool, optional): If true, take in a warped image
                and return an undistorted (dewarped/ideal) image. If false,
                take in an undistorted image and return a warped image.
            image_scale: (float, optional) 0-1 fraction of the original image
                size, used when warping/dewarping downsampled images. See
                pix2face render_img_scale for an example.

        Returns:
            np.ndarray: (I, J, 3) output image
        """

        # Ensure that there is a cached map for these distortion parameters
        dkey = self.distortion_key(camera.distortion_params, image_scale)
        if dkey not in self._maps_ideal_to_warped:
            self.make_distortion_map(camera, inversion_downsample, image_scale)

        if warped_to_ideal:
            inverse_map = self._maps_ideal_to_warped[dkey]
        else:
            inverse_map = self._maps_warped_to_ideal[dkey]

        warped_image = flexible_inputs_warp(
            input_image=input_image,
            inverse_map=inverse_map,
            interpolation_order=interpolation_order,
            fill_value=fill_value,
        )

        return warped_image

    def warp_dewarp_pixels(
        self,
        camera: PhotogrammetryCamera,
        pixels: np.ndarray,
        inversion_downsample: int = 8,
        warped_to_ideal: bool = True,
    ):
        """
        Either apply a camera's distortion model to go from ideal pixels→warped
        or undo the camera's distortion model to go from warped pixels→ideal,
        depending on the warped_to_ideal flag.

        Note that the warp map is cached, so the first call will take longer
        and subsequent calls should be much faster.

        Arguments:
            camera (PhotogrammetryCamera): Camera with parameters that
                define the warp process. These include image size, principal
                point, focal length, and distortion parameters.
            pixels (np.ndarray): (N, 2) Pixels locations in (i, j) format.
            inversion_downsample (int, optional): The distortion map creation
                process is too heavyweight for really high-res images,
                downsampling the inversion process gets a similar result with
                less computation.
            warped_to_ideal (bool, optional): If true, take in a warped image
                and return an undistorted (dewarped/ideal) image. If false,
                take in an undistorted image and return a warped image.

        Returns:
            np.ndarray: (N, 2) warped/dewarped output pixel locations (i, j)
                Note that the output is floating point (subpixel)
        """

        # Ensure that there is a cached map for these distortion parameters
        dkey = self.distortion_key(camera.distortion_params)
        if dkey not in self._maps_ideal_to_warped:
            self.make_distortion_map(camera, inversion_downsample)

        # Get the right mapping array
        if warped_to_ideal:
            rowmap, colmap = self._maps_warped_to_ideal[dkey]
        else:
            rowmap, colmap = self._maps_ideal_to_warped[dkey]

        # Look up the pixel locations
        rows = rowmap[pixels[:, 0], pixels[:, 1]]
        cols = colmap[pixels[:, 0], pixels[:, 1]]
        return np.stack([rows, cols], axis=0).T

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
        detector: Union[RegionDetectionSegmentor, TabularRectangleSegmentor],
        ray_length_meters: float = 1e3,
        boundaries: Optional[Tuple[pv.PolyData, pv.PolyData]] = None,
        limit_ray_length_meters: Optional[float] = None,
        limit_angle_from_vert: Optional[float] = None,
        similarity_threshold_meters: float = 0.1,
        transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        louvain_resolution: float = 1.0,
        out_dir: Optional[PATH_TYPE] = None,
    ) -> np.ndarray:
        """Take per-image detections and triangulate them to 3D locations.

        Args:
            detector (Union[RegionDetectionSegmentor, TabularRectangleSegmentor]):
                Produces detections per image using the get_detection_centers method.
            ray_length_meters (float, optional):
                The length of the visualized rays in meters. Defaults to 1000.
            boundaries (Optional[Tuple[pv.PolyData, pv.PolyData]])
                Defaults to None.
            limit_ray_length_meters (Optional[float])
                Defaults to None.
            limit_angle_from_vert (Optional[float])
                Defaults to None.
            similarity_threshold_meters (float, optional):
                Consider rays a potential match if the distance between them is less than this
                value. Defaults to 0.1.
            transform (Optional[Callable[[np.ndarray], np.ndarray]]):
                Defaults to None.
            louvain_resolution (float, optional):
                The resolution hyperparameter of the networkx.louvain_communities function.
                Height value leads to more communities. Defaults to 1.0.
            out_dir: Optional[PATH_TYPE]
                Defaults to None.

        Returns (np.ndarray):
            (N unique objects, 3), the 3D locations of the identified objects.
            If transform_to_epsg_4978 is None, then this is in the local coordinate
            system of the mesh. If the transform is not None, (lat, lon, alt) is returned.
        """

        # Enforce Path type
        if out_dir is not None:
            out_dir = Path(out_dir)

        def check_exists(file: Union[str, Path]):
            """Helper function to aid in caching and loading cached files."""
            if out_dir is None:
                return False
            if isinstance(file, str):
                path = out_dir / file
            else:
                path = file
            return path.is_file()

        # Determine scale factor relating meters to internal coordinates
        transform_to_epsg_4978 = self.get_local_to_epsg_4978_transform()
        meters_to_local_scale = 1 / get_scale_from_transform(transform_to_epsg_4978)
        ray_length_local = ray_length_meters * meters_to_local_scale
        similarity_threshold_local = similarity_threshold_meters * meters_to_local_scale
        if limit_ray_length_meters is None:
            limit_ray_length_local = None
        else:
            limit_ray_length_local = limit_ray_length_meters * meters_to_local_scale

        # Create line segments emanating from the cameras
        if check_exists("line_segments.npz"):
            line_results = out_dir / "line_segments.npz"
        else:
            line_results = self.calc_line_segments(
                detector=detector,
                boundaries=boundaries,
                ray_length_local=ray_length_local,
                out_dir=out_dir,
                limit_ray_length_local=limit_ray_length_local,
                limit_angle_from_vert=limit_angle_from_vert,
            )
        # Load the results into memory if they were saved to file
        if check_exists(line_results):
            line_results = np.load(line_results)

        # Turn line segments into graph distances, where "close enough"
        # lines are connected nodes in the graph.
        if check_exists("edge_weights.json"):
            weight_results = out_dir / "edge_weights.json"
        else:
            weight_results = calc_graph_weights(
                starts=line_results["ray_starts"],
                ends=line_results["ray_ends"],
                ray_IDs=line_results["ray_IDs"],
                similarity_threshold=similarity_threshold_local,
                out_dir=out_dir,
                step=5000,
                transform=transform,
            )
        # Load the results into memory if they were saved to file
        if check_exists(weight_results):
            weight_results = json.load(weight_results.open("r"))

        # Calculate community identities among the graph weights, where
        # hopefully a preponderance of close line segments indicate a
        # single object detected multiple times.
        if check_exists("communities.npz"):
            community_results = out_dir / "communities.npz"
        else:
            community_results = calc_communities(
                starts=line_results["ray_starts"],
                ends=line_results["ray_ends"],
                edge_weights=weight_results,
                louvain_resolution=louvain_resolution,
                out_dir=out_dir,
                transform_to_epsg_4978=transform_to_epsg_4978,
            )
        # Load the results into memory if they were saved to file
        if check_exists(community_results):
            community_results = np.load(community_results)

        # Return the 3D locations of the community points, preferentially
        # in lat/lon form if it exists.
        return community_results.get(
            "community_points_latlon",
            community_results["community_points"],
        )

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

    def get_vis_mesh(self, frustum_scale: float = 0.1) -> pv.PolyData:
        """Get all the cameras as a mesh representation.

        Args:
            frustum_scale (float, optional): Size of cameras in world units.

        Returns: (PolyData) mesh representation of all cameras as frustums
        """
        return pv.merge(
            [
                camera.get_vis_mesh(frustum_scale=frustum_scale)
                for camera in self.cameras
            ]
        )

    def calc_line_segments(
        self,
        detector: Union[RegionDetectionSegmentor, TabularRectangleSegmentor],
        boundaries: Optional[Tuple[pv.PolyData, pv.PolyData]] = None,
        ray_length_local: float = 1e3,
        out_dir: Optional[PATH_TYPE] = None,
        limit_ray_length_local: Optional[float] = None,
        limit_angle_from_vert: Optional[float] = None,
    ) -> Union[Dict[str, np.ndarray], Path]:
        """
        For each camera in the set, this method:
        1. Gets detection centers from the provided detector
        2. Projects rays from the camera through these detection points
        3. Optionally filters rays by angle from vertical
        4. Optionally clips rays to intersection with boundary surfaces
        5. Returns resulting line segments or saves them to file

        Args:
            detector: Detector that provides detection centers for each image
            boundaries: Optional tuple of (upper, lower) boundary surfaces as PyVista PolyData.
                If provided, rays will be clipped to these surfaces.
            ray_length_local: Length of the initial rays in local coordinate units. Default 1e3.
            out_dir: Directory to save the output NPZ file containing ray data. If this is None,
                the data will be returned as a dict. If this is a path, "line_segments.npz"
                will be saved in that directory. Default is None.
            limit_ray_length_local: Optional max ray length from origin to second boundary.
                This is to mimic measuring from a camera (hypothetical ray source) to
                the ground (assuming the boundaries are given as [ceiling, floor]).
                Default is None, meaning no limit.
            limit_angle_from_vert: Optional max angle (in radians) from vertical.
                Default is None, meaning no limit.

        Returns:
            If out_dir is None, returns a dictionary with:
                - "ray_starts": (N, 3) array of ray start points
                - "ray_ends": (N, 3) array of ray end points
                - "ray_IDs": (N,) array of camera indices that generated each ray
            If out_dir is provided, saves this data to "line_segments.npz" in that directory
        """

        # Record the lines corresponding to each detection and the associated image ID
        all_line_segments = []
        all_image_IDs = []

        # Iterate over the cameras
        for camera_ind in tqdm(
            range(len(self.cameras)), desc="Building line segments per camera"
        ):
            # Get the image filename
            image_filename = str(self.get_image_filename(camera_ind))
            # Get the centers of associated detection from the detector
            # TODO, this only works with "detectors" that can look up the detections based on the
            # filename alone. In the future we might want to support real detectors that actually
            # use the image.
            detection_centers_pixels = detector.get_detection_centers(image_filename)
            # Project rays given the locations of the detections in pixel coordinates
            if len(detection_centers_pixels) > 0:
                # Record the line segments, which will be ordered as alternating (start, end) rows
                line_segments = self.cameras[camera_ind].cast_rays(
                    pixel_coords_ij=detection_centers_pixels,
                    line_length=ray_length_local,
                )
                all_line_segments.append(line_segments)
                # Record which image ID generated each line
                all_image_IDs.append(
                    np.full(
                        int(line_segments.shape[0] / 2),
                        fill_value=camera_ind,
                    )
                )

        if len(all_line_segments) > 0:
            # Concatenate the lists of arrays into a single array
            all_line_segments = np.concatenate(all_line_segments, axis=0)
            all_image_IDs = np.concatenate(all_image_IDs, axis=0)

            # Get the starts and ends, which are alternating rows
            ray_starts = all_line_segments[0::2]
            ray_ends = all_line_segments[1::2]
            # Determine the direction
            ray_directions = ray_ends - ray_starts
            # Make the ray directions unit length
            ray_directions = ray_directions / np.linalg.norm(
                ray_directions, axis=1, keepdims=True
            )

            # Filter by angle from vertical if requested
            if limit_angle_from_vert is not None:
                # Angle from vertical (z-axis): arccos(|z component of unit vector|)
                z_axis = ray_directions[:, 2]
                angles = np.arccos(np.abs(z_axis))
                keep_mask = angles <= limit_angle_from_vert
                ray_starts = ray_starts[keep_mask]
                ray_ends = ray_ends[keep_mask]
                ray_directions = ray_directions[keep_mask]
                all_image_IDs = all_image_IDs[keep_mask]

            if boundaries is not None:
                print("Clipping all line segments to boundary surfaces")
                ray_starts, ray_ends, ray_directions, all_image_IDs = (
                    clip_line_segments(
                        boundaries=boundaries,
                        origins=ray_starts,
                        directions=ray_directions,
                        image_indices=all_image_IDs,
                        ray_limit=limit_ray_length_local,
                    )
                )

        else:
            ray_starts = np.empty((0, 3))
            ray_ends = np.empty((0, 3))
            all_image_IDs = np.empty((0,), dtype=int)

        # Return or save to file
        data = {
            "ray_starts": ray_starts,
            "ray_ends": ray_ends,
            "ray_IDs": all_image_IDs,
        }
        if out_dir is None:
            return data
        else:
            path = Path(out_dir) / "line_segments.npz"
            np.savez(path, **data)
            return path
