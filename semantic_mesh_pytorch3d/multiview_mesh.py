import numpy as np
import pyproj
import pyvista as pv
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
from tqdm import tqdm

from semantic_mesh_pytorch3d.config import PATH_TYPE


class MultiviewMesh:
    def __init__(
        self, mesh_filename: PATH_TYPE, downsample_target: float = 1.0, **kwargs
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
        self.downsample_target = downsample_target

        self.pyvista_mesh = None
        self.pytorch_mesh = None
        self.verts = None
        self.faces = None

        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")

        self.load_mesh(downsample_target=downsample_target)
        self.create_texture(**kwargs)

    def load_mesh(
        self,
        downsample_target: float = 1.0,
    ):
        """Load the pyvista mesh and create the pytorch3d texture

        Args:
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

    def create_texture(self):
        # Abstract method
        raise NotImplementedError()

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

        # Overwrite existing vertices
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

    def texture_with_binary_mask(
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

    def vis(self, show_cameras: bool = False, filename=None):
        """Show the mesh and cameras"""
        plotter = pv.Plotter(off_screen=(filename is not None))
        plotter.add_mesh(self.pyvista_mesh, rgb=True)
        if show_cameras:
            self.camera_set.vis(plotter, add_orientation_cube=True)
        plotter.show(screenshot=filename)

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

    def aggregate_viewpoints_pytorch3d(self):
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
