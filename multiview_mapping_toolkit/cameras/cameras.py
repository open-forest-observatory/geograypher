import os
import shutil
from copy import deepcopy
from pathlib import Path
from typing import List, Tuple, Union

import geopandas as gpd
import numpy as np
import numpy.ma as ma
import pyproj
import pyvista as pv
import torch
from pytorch3d.renderer import PerspectiveCameras
from pyvista import demos
from shapely import Point
from skimage.io import imread
from skimage.transform import resize
from tqdm import tqdm

from multiview_mapping_toolkit.config import PATH_TYPE
from multiview_mapping_toolkit.utils.geospatial import get_projected_CRS
from multiview_mapping_toolkit.utils.image import get_GPS_exif


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
        distortion_params: dict = {},
        lon_lat: Union[None, Tuple[float, float]] = None,
    ):
        """Represents the information about one camera location/image as determined by photogrammetry

        Args:
            image_filename (PATH_TYPE): The image used for reconstruction
            transform (np.ndarray): A 4x4 transform representing the camera-to-world transform
            f (float): Focal length in pixels
            cx (float): Principle point x (pixels)
            cy (float): Principle point y (pixels)
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

        if lon_lat is None:
            self.lon_lat = (None, None)
        else:
            self.lon_lat = lon_lat

        self.image_size = (image_height, image_width)
        self.image = None
        self.cache_image = (
            False  # Only set to true if you can hold all images in memory
        )

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

    def get_pytorch3d_camera(self, device: str):
        """Return a pytorch3d camera based on the parameters from metashape

        Args:
            device (str): What device (cuda/cpu) to put the object on

        Returns:
            pytorch3d.renderer.PerspectiveCameras:
        """
        rotation_about_z = np.array(
            [[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        )
        # Rotate about the Z axis because the NDC coordinates are defined X: left, Y: up and we use X: right, Y: down
        # See https://pytorch3d.org/docs/cameras
        transform_4x4_world_to_cam = rotation_about_z @ self.world_to_cam_transform

        R = torch.Tensor(np.expand_dims(transform_4x4_world_to_cam[:3, :3].T, axis=0))
        T = torch.Tensor(np.expand_dims(transform_4x4_world_to_cam[:3, 3], axis=0))

        # The image size is (height, width) which completely disreguards any other conventions they use...
        image_size = ((self.image_height, self.image_width),)
        # These parameters are in screen (pixel) coordinates.
        # TODO see if a normalized version is more robust for any reason
        fcl_screen = (self.f,)
        prc_points_screen = (
            (self.image_width / 2 + self.cx, self.image_height / 2 + self.cy),
        )

        # Create camera
        # TODO use the pytorch3d FishEyeCamera model that uses distortion
        # https://pytorch3d.readthedocs.io/en/latest/modules/renderer/fisheyecameras.html?highlight=distortion
        cameras = PerspectiveCameras(
            R=R,
            T=T,
            focal_length=fcl_screen,
            principal_point=prc_points_screen,
            device=device,
            in_ndc=False,  # screen coords
            image_size=image_size,
        )
        return cameras

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
        scaled_cy = self.cx / self.f

        right = scaled_cx + scaled_halfwidth
        left = scaled_cx - scaled_halfwidth
        top = scaled_cy + scaled_halfheight
        bottom = scaled_cy - scaled_halfheight

        vertices = np.array(
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
                [3, 3, 4, 1],  # endcap tiangle #1
            ]
        )
        # All blue except the top surface is red
        face_colors = np.array(
            [[0, 0, 1], [0, 0, 1], [0, 0, 1], [1, 0, 0], [0, 0, 1], [0, 0, 1]]
        ).astype(float)

        # Create a mesh for the camera frustum
        frustum = pv.PolyData(projected_vertices[:3].T, faces)
        # Unsure exactly what's going on here, but it's required for it to be valid
        frustum.triangulate()
        # Show the mesh with the given face colors
        # TODO understand how this understands it's face vs. vertex colors? Simply by checking the number of values?
        plotter.add_mesh(frustum, scalars=face_colors, rgb=True)


class PhotogrammetryCameraSet:
    f = None
    cx = None
    cy = None
    image_width = None
    image_height = None

    def __init__(self, camera_file: PATH_TYPE, image_folder: PATH_TYPE, **kwargs):
        """
        Create a camera set from a metashape .xml camera file and the path to the image folder


        Args:
            camera_file (PATH_TYPE): Path to the .xml camera export from Metashape
            image_folder (PATH_TYPE): Path to the folder of images used by Metashape
        """
        self.image_folder = image_folder
        self.camera_file = camera_file

        (
            self.image_filenames,
            self.cam_to_world_transforms,
            self.sensor_IDs,
            self.lan_lats,
            self.sensors_dict,
        ) = self.parse_input(
            camera_file=camera_file, image_folder=image_folder, **kwargs
        )

        missing_images = self.find_mising_images()
        if len(missing_images) > 0:
            print(missing_images)
            raise ValueError("Missing images displayed above")

        self.cameras = []

        for image_filename, cam_to_world_transform, sensor_ID, lon_lat in zip(
            self.image_filenames,
            self.cam_to_world_transforms,
            self.sensor_IDs,
            self.lon_lats,
        ):
            sensor_params = self.sensors_dict[sensor_ID]
            new_camera = PhotogrammetryCamera(
                image_filename, cam_to_world_transform, lon_lat=lon_lat, **sensor_params
            )
            self.cameras.append(new_camera)

    def find_mising_images(self):
        invalid_images = []
        for image_file in self.image_filenames:
            if not image_file.is_file():
                invalid_images.append(image_file)

        return invalid_images

    def parse_input(self, camera_file: PATH_TYPE, image_folder: PATH_TYPE):
        """
        Parse the software-specific camera files and populate required member fields.

        Args:
            camera_file (PATH_TYPE): Path to the camera file
            image_folder (PATH_TYPE): Path to the root of the image folder
        """
        raise NotImplementedError("Abstract base class")

    def n_cameras(self) -> int:
        """Return the number of cameras"""
        return len(self.cameras)

    def n_image_channels(self) -> int:
        """Return the number of channels in the image"""
        return 3

    def get_camera_by_index(self, index: int) -> PhotogrammetryCamera:
        if index >= len(self.cameras):
            raise ValueError("Requested camera ind larger than list")
        return self.cameras[index]

    def get_subset_cameras(self, inds: List[int]):
        subset_camera_set = deepcopy(self)
        subset_camera_set.cameras = [self.cameras[i] for i in inds]
        return subset_camera_set

    def get_image_by_index(self, index: int, image_scale: float = 1.0) -> np.ndarray:
        return self.get_camera_by_index(index).get_image(image_scale=image_scale)

    def get_image_filename(self, index: int, absolute=False):
        filename = self.get_camera_by_index(index).image_filename
        if not absolute:
            filename = Path(filename).relative_to(self.image_folder)
        return filename

    def get_pytorch3d_camera(self, device: str):
        """
        Return a pytorch3d cameras object based on the parameters from metashape.
        This has the information from each of the camears in the set to enabled batched rendering.


        Args:
            device (str): What device (cuda/cpu) to put the object on

        Returns:
            pytorch3d.renderer.PerspectiveCameras:
        """
        # Get the pytorch3d cameras for each of the cameras in the set
        p3d_cameras = [camera.get_pytorch3d_camera(device) for camera in self.cameras]
        # Get the image sizes
        image_sizes = [camera.image_size.cpu().numpy() for camera in p3d_cameras]
        # Check that all the image sizes are the same because this is required for proper batched rendering
        if np.any([image_size != image_sizes[0] for image_size in image_sizes]):
            raise ValueError("Not all cameras have the same image size")
        # Create the new pytorch3d cameras object with the information from each camera
        cameras = PerspectiveCameras(
            R=torch.cat([camera.R for camera in p3d_cameras], 0),
            T=torch.cat([camera.T for camera in p3d_cameras], 0),
            focal_length=torch.cat([camera.focal_length for camera in p3d_cameras], 0),
            principal_point=torch.cat(
                [camera.get_principal_point() for camera in p3d_cameras], 0
            ),
            device=device,
            in_ndc=False,  # screen coords
            image_size=image_sizes[0],
        )
        return cameras

    def save_images(self, output_folder, copy=False):
        for i in tqdm(range(len(self.cameras))):
            output_file = Path(
                output_folder, self.get_image_filename(i, absolute=False)
            )
            output_file.parent.mkdir(parents=True, exist_ok=True)
            src_file = self.get_image_filename(i, absolute=True)
            if copy:
                shutil.copy(src_file, output_file)
            else:
                os.symlink(src_file, output_file)

    def get_lon_lat_coords(self):
        """Returns a list of GPS coords for each camera"""
        return [
            x.get_lon_lat()
            for x in tqdm(self.cameras, desc="Loading GPS data for camera set")
        ]

    def get_subset_near_geofile(
        self, geofile: PATH_TYPE, buffer_radius_meters: float = 50
    ):
        """Return cameras that are within a radius of the provided geometry

        Args:
            geofile (PATH_TYPE): Path to a geofile readable by geopandas
            buffer_radius_meters (float, optional): Return points within this buffer of the geometry. Defaults to 50.
        """
        # Read in the geofile
        geodata = gpd.read_file(geofile)
        # Transform to the local geometric CRS if it's lat lon
        if geodata.crs == pyproj.CRS.from_epsg(4326):
            point = geodata["geometry"][0].centroid
            geometric_crs = get_projected_CRS(lon=point.x, lat=point.y)
            geodata.to_crs(geometric_crs, inplace=True)
        # Merge all of the elements together into one multipolygon, destroying any attributes that were there
        geodata = geodata.dissolve()
        # Expand the geometry of the shape by the buffer
        geodata["geometry"] = geodata.buffer(buffer_radius_meters)

        # Read the locations of all the points
        # TODO do these need to be swapped
        image_locations = [Point(*x) for x in self.get_lon_lat_coords()]
        # Create a dataframe, assuming inputs are lat lon
        image_locations_df = gpd.GeoDataFrame(
            geometry=image_locations, crs=pyproj.CRS.from_epsg(4326)
        )
        image_locations_df.to_crs(geodata.crs, inplace=True)
        # Add an index row because the normal index will be removed in subsequent operations
        image_locations_df["index"] = image_locations_df.index

        points_in_field_buffer = gpd.sjoin(image_locations_df, geodata, how="left")
        valid_camera_points = np.isfinite(
            points_in_field_buffer["index_right"].to_numpy()
        )
        valid_camera_inds = np.where(valid_camera_points)[0]
        # How to instantiate from a list of cameras

        # Is there a better way? Are there side effects I'm not thinking of?
        subset_camera_set = self.get_subset_cameras(valid_camera_inds)
        return subset_camera_set

    def vis(
        self,
        plotter: pv.Plotter = None,
        add_orientation_cube: bool = False,
        show: bool = False,
        frustum_scale: float = 1,
        force_xvfb: bool = False,
    ):
        """Visualize all the cameras

        Args:
            plotter (pv.Plotter): Plotter to add the cameras to. If None, will be created and then plotted
            add_orientation_cube (bool, optional): Add a cube to visualize the coordinate system. Defaults to False.
            show (bool, optional): Show the results instead of waiting for other content to be added
            frustum_scale (float, optional): Size of cameras in world units
            force_xvfb (bool, optional): Force a headless rendering backend
        """

        if plotter is None:
            plotter = pv.Plotter()
            show = True

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
                pv.start_xvfb()
            plotter.show()