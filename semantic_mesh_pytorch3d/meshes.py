import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import pyvista as pv
import torch
from pytorch3d.io import load_objs_as_meshes
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

from semantic_mesh_pytorch3d.cameras import MetashapeCameraSet


class Pytorch3DMesh:
    def __init__(self, mesh_filename, camera_filename, image_folder):
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
        self.load_mesh()

    def load_mesh(
        self,
        texture_enum=0,
    ):
        # Load the mesh using pyvista
        self.pyvista_mesh = pv.read(self.mesh_filename)

        # Extract the vertices and faces
        verts = self.pyvista_mesh.points
        # See here for format: https://github.com/pyvista/pyvista-support/issues/96
        faces = self.pyvista_mesh.faces.reshape((-1, 4))[:, 1:4]

        self.verts = torch.Tensor(verts.copy()).to(self.device)
        self.faces = torch.Tensor(faces.copy()).to(self.device)

        # Convert RGB values to [0,1] and format correctly
        verts_rgb = torch.Tensor(
            np.expand_dims(self.pyvista_mesh["RGB"] / 255, axis=0)
        ).to(self.device)

        if texture_enum == 0:
            # Create a texture from the colors
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
            self.texture_from_geodata(
                "/ofo-share/repos-david/semantic-mesh-pytorch3d/data/composite_20230520T0519/composite_20230520T0519_crowns.gpkg"
            )
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

    def texture_from_geodata(self, geodata_file):
        raise NotImplementedError()
        # TODO
        gdf = gpd.read_file(geodata_file)

    def vis_pv(self):
        plotter = pv.Plotter(off_screen=False)
        self.camera_set.vis(plotter)
        plotter.add_mesh(self.pyvista_mesh, rgb=True)
        plotter.show(screenshot="vis/render.png")

    def aggregate_numpy(self):
        # Initialize a masked array to record values
        summed_values = ma.array(
            data=np.zeros((self.pyvista_mesh.points.shape[0], 3)),
            mask=np.ones((self.pyvista_mesh.points.shape[0], 3)).astype(bool),
        )

        counts = np.zeros((self.pyvista_mesh.points.shape[0], 3))
        for i in tqdm(range(len(self.camera_set.cameras))):
            filename = self.camera_set.cameras[i].filename
            img = plt.imread(filename)
            colors_per_vertex = self.camera_set.cameras[i].splat_mesh_verts(
                self.pyvista_mesh.points, img
            )
            all_values = ma.stack((summed_values, colors_per_vertex), axis=2)
            summed_values = all_values.sum(axis=2)
            counts[np.logical_not(colors_per_vertex.mask)] = (
                counts[np.logical_not(colors_per_vertex.mask)] + 1
            )
        mean_colors = (summed_values / counts).astype(np.uint8)
        plotter = pv.Plotter()
        plotter.add_mesh(self.pyvista_mesh, scalars=mean_colors, rgb=True)
        plotter.show()

    def render_pytorch3d(self):
        # Render each image individually.
        # TODO this could be accelerated by inteligent batching
        inds = np.arange(len(self.camera_set.cameras))
        np.random.shuffle(inds)

        for i in tqdm(inds):
            # Create a camera from the metashape parameters
            cameras = self.camera_set.cameras[i].get_pytorch3d_camera(self.device)

            # Load the image
            filename = self.camera_set.cameras[i].filename
            img = plt.imread(filename)

            # Create ambient light so it doesn't effect the color
            lights = AmbientLights(device=self.device)

            # Set up the rasterizer
            image_size = img.shape[:2]
            raster_settings = RasterizationSettings(
                image_size=image_size,
                blur_radius=0.0,
                faces_per_pixel=1,
            )

            # Create a Phong renderer by composing a rasterizer and a shader. The textured Phong shader will
            # interpolate the texture uv coordinates for each vertex, sample from a texture image and
            # apply the Phong lighting model
            renderer = MeshRenderer(
                rasterizer=MeshRasterizer(
                    cameras=cameras, raster_settings=raster_settings
                ).to(self.device),
                shader=HardGouraudShader(
                    device=self.device, cameras=cameras, lights=lights
                ),
            ).to(self.device)

            images = renderer(self.pytorch_mesh)
            f, ax = plt.subplots(1, 2)
            rendered = images[0, ..., :3].cpu().numpy()
            rendered = np.flip(rendered, axis=(0, 1))
            ax[0].imshow(rendered)
            ax[1].imshow(img)
            ax[0].set_title("Rendered image")
            ax[1].set_title("Real image")
            plt.savefig(f"vis/pred_{i:03d}.png")
            plt.close()
