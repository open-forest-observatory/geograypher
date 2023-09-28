import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import pyvista as pv
import rasterio as rio
import skimage
import torch
from pytorch3d.renderer import (
    AmbientLights,
    HardGouraudShader,
    MeshRasterizer,
    RasterizationSettings,
    TexturesVertex,
)
from pytorch3d.structures import Meshes
from scipy.spatial.distance import cdist
from shapely.geometry import Point
from tqdm import tqdm

from semantic_mesh_pytorch3d.cameras import MetashapeCameraSet
from semantic_mesh_pytorch3d.config import (
    COLORS,
    DEFAULT_DEM,
    DEFAULT_GEOPOLYGON_FILE,
    PATH_TYPE,
)


class Pytorch3DMesh:
    def __init__(
        self,
        mesh_filename: PATH_TYPE,
        camera_filename: PATH_TYPE,
        image_folder: PATH_TYPE,
        texture_enum: int = 0,
    ):
        """This object handles most of the high-level operations in this project

        Args:
            mesh_filename (PATH_TYPE): Path to mesh in metashape's local coordinate system, .ply type
            camera_filename (PATH_TYPE): Path to the .xml metashape camera output
            image_folder (PATH_TYPE): Path to the folders used for reconstruction
            texture_enum (int, optional): Which type of texture to use. 0 is the real color,
                                          1 is a dummy texture, and 2 is from a geofile. Defaults to 0.
        """
        self.mesh_filename = mesh_filename
        self.image_folder = image_folder

        self.pyvista_mesh = None
        self.pyvista_mesh = None

        self.device = None
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")

        self.camera_set = MetashapeCameraSet(camera_filename, image_folder)
        self.load_mesh(texture_enum, 0.25)

    def load_mesh(
        self,
        texture_enum: int,
        downsample_target: float = 1.0,
    ):
        """Load the pyvista mesh and create the pytorch3d texture

        Args:
            texture_enum (int): Which type of texture to use
            downsample_target (float, optional):
                What fraction of mesh vertices to downsample to. Defaults to 1.0, (does nothing).

        Raises:
            ValueError: Invalid texture enum
        """
        # Load the mesh using pyvista
        self.pyvista_mesh = pv.read(self.mesh_filename)
        # Downsample mesh if needed
        if downsample_target != 1.0:
            # TODO try decimate_pro and compare quality and runtime
            # TODO see if there's a way to preserve the mesh colors
            # TODO also see this decimation algorithm: https://pyvista.github.io/fast-simplification/
            self.pyvista_mesh = self.pyvista_mesh.decimate(
                target_reduction=(1 - downsample_target)
            )

        # Extract the vertices and faces
        verts = self.pyvista_mesh.points
        # See here for format: https://github.com/pyvista/pyvista-support/issues/96
        faces = self.pyvista_mesh.faces.reshape((-1, 4))[:, 1:4]

        self.verts = torch.Tensor(verts.copy()).to(self.device)
        self.faces = torch.Tensor(faces.copy()).to(self.device)

        if texture_enum == 0:
            # Create a texture from the colors
            # Convert RGB values to [0,1] and format correctly
            if "RGB" in self.pyvista_mesh.array_names:
                verts_rgb = torch.Tensor(
                    np.expand_dims(self.pyvista_mesh["RGB"] / 255, axis=0)
                ).to(self.device)
            else:
                # Default gray color
                print(self.pyvista_mesh.array_names)
                verts_rgb = torch.Tensor(
                    np.full((1, self.pyvista_mesh.n_points, 3), 0.5)
                ).to(self.device)
            textures = TexturesVertex(verts_features=verts_rgb).to(self.device)
            self.pytorch_mesh = Meshes(
                verts=[self.verts], faces=[self.faces], textures=textures
            )
            self.pytorch_mesh = self.pytorch_mesh.to(self.device)
        elif texture_enum == 1:
            # Create a dummy texture
            self.dummy_texture()
        elif texture_enum == 2:
            # Create a texture from a geofile
            self.geodata_texture()
        elif texture_enum == 3:
            # Create a texture from a geofile
            self.height_threshold()
        else:
            raise ValueError(f"Invalide texture enum {texture_enum}")

    def transform_vertices(self, transform_4x4: np.ndarray, in_place: bool = False):
        """Apply a transform to the vertex coordinates

        Args:
            transform_4x4 (np.ndarray): Transform to be applied
            in_place (bool): Should the vertices be updated
        """
        homogenous_local_points = np.vstack(
            (self.pyvista_mesh.points.T, np.ones(self.pyvista_mesh.n_points))
        )
        transformed_local_points = transform_4x4 @ homogenous_local_points
        transformed_local_points = transformed_local_points[:3].T

        if in_place:
            self.pyvista_mesh.points = transformed_local_points.copy()
        return transformed_local_points

    def get_vertices_in_CRS(self, output_CRS: pyproj.CRS):
        """Return the coordinates of the mesh vertices in a given CRS

        Args:
            output_CRS (pyproj.CRS): The coordinate reference system to transform to

        Returns:
            np.ndarray: (n_points, 3)
        """
        # The mesh points are defined in an arbitrary local coordinate system but we can transform them to EPGS:4978,
        # the earth-centered, earth-fixed coordinate system, using an included transform
        epgs4978_verts = self.transform_vertices(
            self.camera_set.local_to_epgs_4978_transform
        )

        output_CRS = pyproj.CRS.from_epsg(output_CRS.to_epsg())
        # Build a pyproj transfrormer from EPGS:4978 to the desired CRS
        transformer = pyproj.Transformer.from_crs(
            pyproj.CRS.from_epsg(4978), output_CRS
        )

        # Transform the coordinates
        verts_in_output_CRS = transformer.transform(
            xx=epgs4978_verts[:, 0],
            yy=epgs4978_verts[:, 1],
            zz=epgs4978_verts[:, 2],
        )
        # Stack and transpose
        verts_in_output_CRS = np.vstack(verts_in_output_CRS).T

        return verts_in_output_CRS

    def color_with_binary_mask(
        self,
        binary_mask: np.ndarray,
        color_true: list,
        color_false: list,
        vis: bool = False,
    ):
        """Color the pyvista and pytorch3d meshes based on a binary mask and two colors

        Args:
            binary_mask (np.ndarray): Mask to differentiate the two colors
            color_true (list): Color for points corresponding to "true" in the mask
            color_false (list): Color for points corresponding to "false" in the mask
            vis (bool, optional): Show the colored mesh. Defaults to False.
        """
        # Fill the colors with the background color
        colors_tensor = (
            (torch.Tensor([color_false]))
            .repeat(self.pyvista_mesh.points.shape[0], 1)
            .to(self.device)
        )
        # create the forgound color
        true_color_tensor = (torch.Tensor([color_true])).to(self.device)
        # Set the indexed points to the forground color
        colors_tensor[binary_mask] = true_color_tensor
        if vis:
            self.pyvista_mesh["colors"] = colors_tensor.cpu().numpy()
            self.pyvista_mesh.plot(rgb=True, scalars="colors")

        # Color pyvista mesh
        self.pyvista_mesh["RGB"] = colors_tensor.cpu().numpy()

        # Add singleton batch dimension so it is (1, n_verts, 3)
        colors_tensor = torch.unsqueeze(colors_tensor, 0)

        # Create a pytorch3d texture and add it to the mesh
        textures = TexturesVertex(verts_features=colors_tensor)
        self.pytorch_mesh = Meshes(
            verts=[self.verts], faces=[self.faces], textures=textures
        )

    def dummy_texture(self, use_colorseg: bool = True):
        """Create a dummy texture for debuging

        Args:
            use_colorseg (bool, optional):
                Segment based on color into two classes
                Otherwise, segment based on a centered circle. Defaults to True.
        """
        green = [34, 139, 34]
        orange = [255, 165, 0]
        RGB_values = self.pyvista_mesh["RGB"]
        if use_colorseg:
            test_values = np.array([green, orange])
            dists = cdist(RGB_values, test_values)
            inds = np.argmin(dists, axis=1)
        else:
            XYZ_values = self.pyvista_mesh.points
            center = np.mean(XYZ_values, axis=0, keepdims=True)
            dists_to_center = np.linalg.norm(XYZ_values[:, :2] - center[:, :2], axis=1)
            cutoff_value = np.quantile(dists_to_center, [0.1])
            inds = (dists_to_center > cutoff_value).astype(int)
        dummy_RGB_values = np.zeros_like(RGB_values)
        dummy_RGB_values[inds == 0] = np.array(green)
        dummy_RGB_values[inds == 1] = np.array(orange)
        verts_rgb = torch.Tensor(np.expand_dims(dummy_RGB_values / 255, axis=0)).to(
            self.device
        )  # (1, V, 3)
        textures = TexturesVertex(verts_features=verts_rgb.to(self.device))
        self.pytorch_mesh = Meshes(
            verts=[self.verts], faces=[self.faces], textures=textures
        )

    def geodata_texture(
        self,
        geo_polygon_file: PATH_TYPE = DEFAULT_GEOPOLYGON_FILE,
        vis: bool = False,
    ):
        """Create a texture from a geofile containing polygons

        Args:
            geo_polygon_file (PATH_TYPE, optional):
                Filepath to read from. Must be able to be opened by geopandas. Defaults to DEFAULT_GEOPOLYGON_FILE.
            vis (bool, optional): Show the texture. Defaults to False.
        """
        # Read the polygon data about the tree crown segmentation
        geo_polygons = gpd.read_file(geo_polygon_file)
        # Get the vertices in the same CRS as the geofile
        verts_in_geopolygon_crs = self.get_vertices_in_CRS(geo_polygons.crs)

        # Taken from https://www.matecdev.com/posts/point-in-polygon.html
        # Convert points into georeferenced dataframe
        # TODO consider removing this line since it's not strictly needed
        df = pd.DataFrame(
            {
                "east": verts_in_geopolygon_crs[:, 0],
                "north": verts_in_geopolygon_crs[:, 1],
            }
        )
        df["coords"] = list(zip(df["east"], df["north"]))
        df["coords"] = df["coords"].apply(Point)
        points = gpd.GeoDataFrame(df, geometry="coords", crs=geo_polygons.crs)
        # Add an index column because the normal index will not be preserved in future operations
        points["id"] = df.index

        # Select points that are within the polygons
        points_in_polygons = gpd.tools.overlay(points, geo_polygons, how="intersection")
        # Create an array corresponding to all the points and initialize to NaN
        polygon_IDs = np.full(shape=points.shape[0], fill_value=np.nan)
        # Assign points that are inside a given tree with that tree's ID
        polygon_IDs[points_in_polygons["id"].to_numpy()] = points_in_polygons[
            "treeID"
        ].to_numpy()

        # These points are within a tree
        # TODO, in the future we might want to do something more sophisticated than tree/not tree
        inside_tree_polygon = np.isfinite(polygon_IDs)
        above_ground = self.get_height_above_ground() > 2
        is_tree = (
            torch.Tensor(np.logical_and(inside_tree_polygon, above_ground))
            .to(self.device)
            .to(torch.bool)
        )

        self.color_with_binary_mask(
            is_tree,
            color_true=np.array(COLORS["canopy"]) / 255.0,
            color_false=np.array(COLORS["earth"]) / 255.0,
        )

    def get_height_above_ground(self, DEM_file: PATH_TYPE = DEFAULT_DEM):
        """Compute the height above groun for each point on the mesh

        Args:
            DEM_file (PATH_TYPE, optional): The path the the DEM/DTM file from metashape. Defaults to DEFAULT_DEM.

        Returns:
            np.ndarray: Heights above the ground for each point, aranged in the same order as mesh points (meters)
        """
        # Open the DEM file
        DEM = rio.open(DEM_file)
        # Get the mesh points in the coordinate reference system of the DEM
        verts_in_DEM_crs = self.get_vertices_in_CRS(DEM.crs)

        x_points = verts_in_DEM_crs[:, 1].tolist()
        y_points = verts_in_DEM_crs[:, 0].tolist()
        point_elevation_meters = verts_in_DEM_crs[:, 2]

        zipped_points = zip(x_points, y_points)
        DEM_elevation_meters = np.squeeze(np.array(list(DEM.sample(zipped_points))))

        # We want to find a height about the ground by subtracting the DEM from the
        # mesh points
        height_above_ground = point_elevation_meters - DEM_elevation_meters
        return height_above_ground

    def height_threshold(self, DEM_file: PATH_TYPE = DEFAULT_DEM, threshold=2):
        """Texture by thresholding the height above groun

        Args:
            DEM_file (PATH_TYPE, optional): Filepath for DEM/DTM file from metashape. Defaults to DEFAULT_DEM.
            threshold (int, optional): Height above gound to be considered not ground (meters). Defaults to 2.
        """
        # Get the height of each mesh point above the ground
        height_above_ground = self.get_height_above_ground(DEM_file=DEM_file)
        # Threshold to dermine if it's ground or not
        ground_points = height_above_ground < threshold
        # Color the mesh with this mask
        self.color_with_binary_mask(
            ground_points,
            color_true=np.array(COLORS["earth"]) / 255.0,
            color_false=np.array(COLORS["canopy"]) / 255.0,
        )

    def vis(self):
        """Show the mesh and cameras"""
        plotter = pv.Plotter(off_screen=False)
        self.camera_set.vis(plotter, add_orientation_cube=True)
        plotter.add_mesh(self.pyvista_mesh, rgb=True)
        plotter.show(screenshot="vis/render.png")

    def aggregate_viewpoints_naive(self):
        """
        Aggregate the information from all images onto the mesh without considering occlusion
        or distortion parameters
        """
        # Initialize a masked array to record values
        summed_values = np.zeros((self.pyvista_mesh.points.shape[0], 3))

        counts = np.zeros((self.pyvista_mesh.points.shape[0], 3))
        for i in tqdm(range(len(self.camera_set.cameras))):
            # This is actually the bottleneck in the whole process
            img = skimage.io.imread(self.camera_set.cameras[i].image_filename)
            colors_per_vertex = self.camera_set.cameras[i].project_mesh_verts(
                self.pyvista_mesh.points, img, device=self.device
            )
            summed_values = summed_values + colors_per_vertex.data
            counts[np.logical_not(colors_per_vertex.mask)] = (
                counts[np.logical_not(colors_per_vertex.mask)] + 1
            )
        mean_colors = (summed_values / counts).astype(np.uint8)
        plotter = pv.Plotter()
        plotter.add_mesh(self.pyvista_mesh, scalars=mean_colors, rgb=True)
        plotter.show()

    def get_rasterization_results(self, camera_ind: int, image_scale: float = 1.0):
        """Use pytorch3d to get correspondences between pixels and vertices

        Args:
            camera_ind (int): Which camera to evaluate
            img_scale (float): How much to resize the image by

        Returns:
            pytorch3d.PerspectiveCamera: The camera corresponding to the index
            pytorch3d.Fragments: The rendering results from the rasterer, before the shader
        """
        # Create a camera from the metashape parameters
        camera = self.camera_set.cameras[camera_ind].get_pytorch3d_camera(self.device)

        # Load the image
        img = skimage.io.imread(self.camera_set.cameras[camera_ind].image_filename)
        if img.dtype == np.uint8:
            img = img / 255.0

        if image_scale != 1.0:
            img = skimage.transform.resize(
                img, (int(img.shape[0] * image_scale), int(img.shape[1] * image_scale))
            )

        # Set up the rasterizer
        image_size = img.shape[:2]
        raster_settings = RasterizationSettings(
            image_size=image_size,
            blur_radius=0.0,
            faces_per_pixel=1,
        )

        # Don't wrap this in a MeshRenderer like normal because we need intermediate results
        rasterizer = MeshRasterizer(cameras=camera, raster_settings=raster_settings).to(
            self.device
        )

        fragments = rasterizer(self.pytorch_mesh)
        return camera, fragments, img

    def aggregate_viepoints_pytorch3d(self):
        """
        Aggregate information from different viepoints onto the mesh faces using pytorch3d.
        This considers occlusions but is fairly slow
        """
        # This is where the colors will be aggregated
        # This should be big enough to not overflow
        face_colors = np.zeros((self.pyvista_mesh.n_faces, 3), dtype=np.uint32)
        counts = np.zeros(self.pyvista_mesh.n_faces, dtype=np.uint8)
        for i in tqdm(range(len(self.camera_set.cameras))):
            _, fragments, img = self.get_intermediate_rendering_results(i)
            # Set up inds
            if i == 0:
                inds = np.meshgrid(
                    np.arange(img.shape[0]), np.arange(img.shape[1]), indexing="ij"
                )
                flat_i_inds = inds[0].flatten()
                flat_j_inds = inds[1].flatten()

            pix_to_face = fragments.pix_to_face[0, :, :, 0].cpu().numpy().flatten()
            new_colors = np.zeros((self.pyvista_mesh.n_faces, 3), dtype=np.uint32)
            new_colors[pix_to_face] = img[flat_i_inds, flat_j_inds]
            face_colors = face_colors + new_colors
            unique_faces = np.unique(pix_to_face)
            counts[unique_faces] = counts[unique_faces] + 1
        self.pyvista_mesh["face_colors"] = (
            face_colors / np.expand_dims(counts, 1)
        ).astype(np.uint8)
        self.pyvista_mesh.plot(scalars="face_colors", rgb=True)

    def render_pytorch3d(self, image_scale=1.0):
        """Render an image from the viewpoint of each camera"""
        # Render each image individually.
        # TODO this could be accelerated by inteligent batching
        inds = np.arange(len(self.camera_set.cameras))
        np.random.shuffle(inds)

        for i in tqdm(inds):
            # This part is shared across many tasks
            camera, fragments, img = self.get_rasterization_results(
                i, image_scale=image_scale
            )

            # Create ambient light so it doesn't effect the color
            lights = AmbientLights(device=self.device)
            # Create a shader
            shader = HardGouraudShader(
                device=self.device, cameras=camera, lights=lights
            )

            # Render te images using the shader
            images = shader(fragments, self.pytorch_mesh)

            # Extract and save images
            rendered = images[0, ..., :3].cpu().numpy()
            composite = (
                np.clip(np.concatenate((img, rendered, (img + rendered) / 2.0)), 0, 1)
                * 255
            ).astype(np.uint8)
            skimage.io.imsave(f"vis/pred_{i:03d}.png", composite)
