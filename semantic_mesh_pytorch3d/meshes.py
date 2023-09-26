import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import pyvista as pv
import torch
from imageio import imread, imwrite
from scipy.spatial.distance import cdist
from pytorch3d.renderer import (
    MeshRasterizer,
    MeshRenderer,
    RasterizationSettings,
    HardGouraudShader,
    AmbientLights,
    TexturesVertex,
)
from pytorch3d.structures import Meshes
from tqdm import tqdm
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from semantic_mesh_pytorch3d.cameras import MetashapeCameraSet
from semantic_mesh_pytorch3d.config import DEFAULT_GEOREF_MESH, DEFAULT_GEOFILE


class Pytorch3DMesh:
    def __init__(self, mesh_filename, camera_filename, image_folder, texture_enum=0):
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
        self.load_mesh(texture_enum, 0.05)

    def load_mesh(
        self,
        texture_enum,
        downsample_target=1.0,
    ):
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
            self.create_dummy_texture()
        elif texture_enum == 2:
            # Create a texture from a geofile
            self.texture_from_geodata()
        else:
            raise ValueError(f"Invalide texture enum {texture_enum}")

    def create_dummy_texture(self, use_colorseg=True):
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

    def texture_from_geodata(
        self,
        geo_data_file=DEFAULT_GEOFILE,
        geo_mesh=DEFAULT_GEOREF_MESH,
    ):
        # Read the data
        gdf = gpd.read_file(geo_data_file)
        # convert to lat, lon so it matches the mesh
        gdf = gdf.to_crs("EPSG:4326")

        # Load the mesh, assumed to be lat, lon
        g_mesh = pv.read(geo_mesh)
        # Note that lat, lon convention doesn't correspond to how it's said
        lat = np.array(g_mesh.points[:, 1])
        lon = np.array(g_mesh.points[:, 0])
        # Normalize this axis since it's in meters and the rest are lat-lon
        g_mesh.points[:, 2] = g_mesh.points[:, 2] / 111119

        # Taken from https://www.matecdev.com/posts/point-in-polygon.html
        # Convert lat-lon points to GeoDataFrame
        df = pd.DataFrame({"lon": lon, "lat": lat})
        df["coords"] = list(zip(df["lon"], df["lat"]))
        df["coords"] = df["coords"].apply(Point)
        points = gpd.GeoDataFrame(df, geometry="coords", crs=gdf.crs)

        # Add an index column because the normal index will not be preserved
        points["id"] = df.index

        # Select points that are within the polygons
        pointInPolyws = gpd.tools.overlay(points, gdf, how="intersection")
        # Create an array corresponding to all the points
        polygon_IDs = np.full(shape=points.shape[0], fill_value=np.nan)
        # Assign points that are inside a given tree with that tree's ID
        polygon_IDs[pointInPolyws["id"].to_numpy()] = pointInPolyws["treeID"].to_numpy()

        # These points are within a tree
        is_tree = torch.Tensor(np.isfinite(polygon_IDs)).to(self.device).to(torch.bool)

        # Fill the colors with the background color
        colors = (
            (torch.Tensor([[175, 128, 79]]) / 255.0)
            .repeat(g_mesh.points.shape[0], 1)
            .to(self.device)
        )
        # create the forgound color
        forground_color = (torch.Tensor([[34, 139, 34]]) / 255).to(self.device)
        # Set the indexed points to the forground color
        colors[is_tree] = forground_color

        # Add singleton batch dimension so it is (1, n_verts, 3)
        colors = torch.unsqueeze(colors, 0)

        textures = TexturesVertex(verts_features=colors.to(self.device))
        self.pytorch_mesh = Meshes(
            verts=[self.verts], faces=[self.faces], textures=textures
        )

    def vis_pv(self):
        plotter = pv.Plotter(off_screen=False)
        self.camera_set.vis(plotter)
        plotter.add_mesh(self.pyvista_mesh, rgb=True)
        plotter.show(screenshot="vis/render.png")

    def aggregate_viewpoints_naive(self):
        # Initialize a masked array to record values
        summed_values = np.zeros((self.pyvista_mesh.points.shape[0], 3))

        counts = np.zeros((self.pyvista_mesh.points.shape[0], 3))
        for i in tqdm(range(len(self.camera_set.cameras))):
            filename = self.camera_set.cameras[i].filename
            # This is actually the bottleneck in the whole process
            img = imread(filename)
            colors_per_vertex = self.camera_set.cameras[i].splat_mesh_verts(
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

    def get_intermediate_rendering_results(self, camera_ind):
        # Create a camera from the metashape parameters
        camera = self.camera_set.cameras[camera_ind].get_pytorch3d_camera(self.device)

        # Load the image
        filename = self.camera_set.cameras[camera_ind].filename
        img = imread(filename)

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

    def render_pytorch3d(self):
        # Render each image individually.
        # TODO this could be accelerated by inteligent batching
        inds = np.arange(len(self.camera_set.cameras))
        np.random.shuffle(inds)

        for i in tqdm(inds):
            # This part is shared across many tasks
            camera, fragments, img = self.get_intermediate_rendering_results(i)

            # Create ambient light so it doesn't effect the color
            lights = AmbientLights(device=self.device)
            # Create a shader
            shader = HardGouraudShader(
                device=self.device, cameras=camera, lights=lights
            )

            # Render te images using the shader
            images = shader(fragments, self.pytorch_mesh)

            # Extract and save images
            rendered = images[0, ..., :3].cpu().numpy() * 255
            composite = np.clip(
                np.concatenate((img, rendered, (img + rendered) / 2.0)), 0, 255
            ).astype(np.uint8)
            imwrite(f"vis/pred_{i:03d}.png", composite)
