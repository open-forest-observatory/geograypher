from pathlib import Path

import numpy as np
import pyproj
import pyvista as pv
from scipy.datasets import face
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

from multiview_prediction_toolkit.cameras import (
    PhotogrammetryCamera,
    PhotogrammetryCameraSet,
)
from multiview_prediction_toolkit.config import PATH_TYPE, VIS_FOLDER


class TexturedPhotogrammetryMesh:
    def __init__(
        self, mesh_filename: PATH_TYPE, downsample_target: float = 1.0, **kwargs
    ):
        """_summary_

        Args:
            mesh_filename (PATH_TYPE): Path to the mesh, in a format pyvista can read
            downsample_target (float, optional): Downsample to this fraction of vertices. Defaults to 1.0.
        """
        self.mesh_filename = Path(mesh_filename)
        self.downsample_target = downsample_target

        self.pyvista_mesh = None
        self.pytorch_mesh = None
        self.verts = None
        self.faces = None
        self.local_to_epgs_4978_transform = None

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
        require_transform=True,
    ):
        """Load the pyvista mesh and create the pytorch3d texture

        Args:
            downsample_target (float, optional):
                What fraction of mesh vertices to downsample to. Defaults to 1.0, (does nothing).
            require_transform (bool): Does a local-to-global transform file need to be available

        Raises:
            FileNotFoundError: Cannot find texture file
            ValueError: Transform file doesn't have 4x4 matrix
        """
        # First look for the transform file because this is fast
        transform_filename = Path(
            str(self.mesh_filename).replace(self.mesh_filename.suffix, "_transform.csv")
        )
        if transform_filename.is_file():
            self.local_to_epgs_4978_transform = np.loadtxt(
                transform_filename, delimiter=","
            )
            if self.local_to_epgs_4978_transform.shape != (4, 4):
                raise ValueError(
                    f"Transform should be (4,4) but is {self.local_to_epgs_4978_transform.shape}"
                )
        elif require_transform:
            raise FileNotFoundError(
                f"Required transform file {transform_filename} file could not be found"
            )

        # Load the mesh using pyvista
        # TODO see if pytorch3d has faster/more flexible readers. I'd assume no, but it's good to check
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

    def create_texture(self, **kwargs):
        """Texture the mesh, potentially with other information"""
        self.pytorch_mesh = Meshes(
            verts=[self.verts],
            faces=[self.faces],
        )
        self.pytorch_mesh = self.pytorch_mesh.to(self.device)

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
        epgs4978_verts = self.transform_vertices(self.local_to_epgs_4978_transform)

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
        # Wrap the color in a numpy array to avoid warning about "tensor from list of arrays is slow"
        colors_tensor = (
            (torch.Tensor(np.array([color_false])))
            .repeat(self.pyvista_mesh.points.shape[0], 1)
            .to(self.device)
        )
        # create the forgound color
        true_color_tensor = (torch.Tensor(np.array([color_true]))).to(self.device)
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

    def vis(
        self,
        interactive=False,
        camera_set: PhotogrammetryCameraSet = None,
        screenshot_filename: PATH_TYPE = None,
        **plotter_kwargs,
    ):
        """Show the mesh and cameras

        Args:
            off_screen (bool, optional): Show offscreen
            camera_set (PhotogrammetryCameraSet, optional): Cameras to visualize. Defaults to None.
            screenshot_filename (PATH_TYPE, optional): Filepath to save to, will show interactively if None. Defaults to None.
        """
        plotter = pv.Plotter(
            off_screen=(not interactive) or (screenshot_filename is None)
        )

        plotter.add_mesh(self.pyvista_mesh, rgb=True)
        if camera_set is not None:
            camera_set.vis(plotter, add_orientation_cube=True)
        plotter.show(screenshot=screenshot_filename, **plotter_kwargs)

    def aggregate_viewpoints_naive(self, camera_set: PhotogrammetryCameraSet):
        """
        Aggregate the information from all images onto the mesh without considering occlusion
        or distortion parameters

        Args:
            camera_set (PhotogrammetryCameraSet): Camera set to use for aggregation
        """
        # Initialize a masked array to record values
        summed_values = np.zeros((self.pyvista_mesh.points.shape[0], 3))

        counts = np.zeros((self.pyvista_mesh.points.shape[0], 3))
        for i in tqdm(range(len(camera_set.cameras))):
            # This is actually the bottleneck in the whole process
            img = camera_set.get_camera_by_index(i).load_image()
            colors_per_vertex = camera_set.cameras[i].project_mesh_verts(
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

    def get_rasterization_results(
        self, camera: PhotogrammetryCamera, image_scale: float = 1.0
    ):
        """Use pytorch3d to get correspondences between pixels and vertices

        Args:
            camera (PhotogrammetryCamera): Camera to get raster for
            img_scale (float): How much to resize the image by

        Returns:
            pytorch3d.PerspectiveCamera: The camera corresponding to the index
            pytorch3d.Fragments: The rendering results from the rasterer, before the shader
        """

        # Create a camera from the metashape parameters
        p3d_camera = camera.get_pytorch3d_camera(self.device)
        image_size = camera.get_image_size(image_scale=image_scale)
        raster_settings = RasterizationSettings(
            image_size=image_size,
            blur_radius=0.0,
            faces_per_pixel=1,
        )

        # Don't wrap this in a MeshRenderer like normal because we need intermediate results
        rasterizer = MeshRasterizer(
            cameras=p3d_camera, raster_settings=raster_settings
        ).to(self.device)

        fragments = rasterizer(self.pytorch_mesh)
        return p3d_camera, fragments

    def aggregate_viewpoints_pytorch3d(
        self,
        camera_set: PhotogrammetryCameraSet,
        camera_inds=None,
        image_scale: float = 1.0,
    ):
        """
        Aggregate information from different viepoints onto the mesh faces using pytorch3d.
        This considers occlusions but is fairly slow

        Args:
            camera_set (PhotogrammetryCamera): Set of cameras to aggregate
            camera_inds: What images to use
            image_scale (float): Scale images
        """
        # TODO add an option to do this with a lower-res image
        # TODO make this return something meaningful rather than side effects/in place ops

        # This is where the colors will be aggregated
        # This should be big enough to not overflow
        n_channels = camera_set.n_image_channels()
        face_colors = np.zeros((self.pyvista_mesh.n_faces, n_channels), dtype=np.uint32)
        counts = np.zeros(self.pyvista_mesh.n_faces, dtype=np.uint16)

        # Set up indices for indexing into the image
        img_shape = camera_set.get_camera_by_index(0).get_image_size(
            image_scale=image_scale
        )
        inds = np.meshgrid(
            np.arange(img_shape[0]), np.arange(img_shape[1]), indexing="ij"
        )
        flat_i_inds = inds[0].flatten()
        flat_j_inds = inds[1].flatten()

        if camera_inds is None:
            # If camera inds are not defined, do them all in a random order
            camera_inds = np.arange(len(camera_set.cameras))
            np.random.shuffle(camera_inds)

        for i in tqdm(camera_inds):
            # Get the photogrammetry camera
            pg_camera = camera_set.get_camera_by_index(i)
            # Do the expensive step to get pixel-to-vertex correspondences
            _, fragments = self.get_rasterization_results(
                camera=pg_camera, image_scale=image_scale
            )
            # Load the image
            img = camera_set.get_image_by_index(i, image_scale=image_scale)

            ## Aggregate image information using the correspondences
            # Extract the correspondences as a flat array
            pix_to_face = fragments.pix_to_face[0, :, :, 0].cpu().numpy().flatten()
            # Build an array to store the new colors
            new_colors = np.zeros(
                (self.pyvista_mesh.n_faces, n_channels), dtype=np.uint32
            )
            # Index the image to fill this array
            # TODO find a way to do this better if there are multiple pixels per face
            # now that behaviour is undefined, I assume the last on indexed just overrides the previous ones
            new_colors[pix_to_face] = img[flat_i_inds, flat_j_inds]
            # Update the face colors
            face_colors = face_colors + new_colors
            # Find unique face indices because we can't increment multiple times like ths
            unique_faces = np.unique(pix_to_face)
            counts[unique_faces] = counts[unique_faces] + 1

        normalized_face_colors = face_colors / np.expand_dims(counts, 1)
        return normalized_face_colors, face_colors, counts

    def show_face_textures(
        self,
        face_textures: np.ndarray,
        screenshot_file: str = None,
        off_screen: bool = False,
    ):
        """Plot the mesh with a given face texturing

        Args:
            face_textures (np.ndarray): (n_faces, n_channels) or (n_faces,) array of values from (0,1)

        Raises:
            ValueError: Face textures are None
        """
        if face_textures is None:
            raise ValueError("No face textures")

        off_screen = off_screen or (screenshot_file is not None)

        # If it's a scalar plot as such
        if len(face_textures.shape) == 1:
            self.pyvista_mesh["face_colors"] = face_textures
            self.pyvista_mesh.plot(
                scalars="face_colors",
                rgb=False,
                full_screen=True,
                screenshot=screenshot_file,
                off_screen=off_screen,
            )
            return

        # Else, clip or pad to three channels
        if face_textures.shape[1] > 3:
            face_colors = face_textures[:, :3]
        else:
            padding_array = np.zeros(
                (face_textures.shape[0], 3 - face_textures.shape[1])
            )
            face_colors = np.concatenate((face_textures, padding_array), axis=1)

        # Set the face colors
        self.pyvista_mesh["face_colors"] = face_colors
        # Plot
        self.pyvista_mesh.plot(
            scalars="face_colors",
            rgb=True,
            full_screen=True,
            screenshot=screenshot_file,
            off_screen=off_screen,
        )

    def render_pytorch3d(
        self,
        camera_set: PhotogrammetryCameraSet,
        image_scale: float = 1.0,
        camera_index=None,
    ):
        """Render an image from the viewpoint of a single camera

        Args:
            camera_set (PhotogrammetryCameraSet): Camera set to use for rendering
            image_scale (float, optional):
                Multiplier on the real image scale to obtain size for rendering. Lower values
                yield a lower-resolution render but the runtime is quiker. Defaults to 1.0.
            camera_indices (ArrayLike | NoneType, optional): Indices to render. If None, render all in a random order
        """
        # Get the photogrametery camera
        pg_camera = camera_set.get_camera_by_index(camera_index)

        # This part is shared across many tasks
        p3d_camera, fragments = self.get_rasterization_results(
            pg_camera, image_scale=image_scale
        )

        # Create ambient light so it doesn't effect the color
        lights = AmbientLights(device=self.device)
        # Create a shader
        shader = HardGouraudShader(
            device=self.device, cameras=p3d_camera, lights=lights
        )

        # Render te images using the shader
        images = shader(fragments, self.pytorch_mesh)

        # Extract the image
        rendered = images[0].cpu().numpy()
        return rendered

    def visualize_renders_pytorch3d(
        self,
        camera_set: PhotogrammetryCameraSet,
        image_scale=1.0,
        camera_indices=None,
        render_folder="renders",
    ):
        """Render an image from the viewpoint of each specified camera and save a composite

        Args:
            camera_set (PhotogrammetryCameraSet): Camera set to use for rendering
            image_scale (float, optional):
                Multiplier on the real image scale to obtain size for rendering. Lower values
                yield a lower-resolution render but the runtime is quiker. Defaults to 1.0.
            camera_indices (ArrayLike | NoneType, optional): Indices to render. If None, render all in a random order
            render_folder (PATH_TYPE, optional): Save images to this folder within vis. Default "renders"
        """
        # Render each image individually.
        # TODO this could be accelerated by inteligent batching
        if camera_indices is None:
            camera_indices = np.arange(len(camera_set.n_cameras()))
            np.random.shuffle(camera_indices)

        save_folder = Path(VIS_FOLDER, render_folder)
        save_folder.mkdir(parents=True, exist_ok=True)

        for i in tqdm(camera_indices):
            rendered = self.render_pytorch3d(
                camera_set=camera_set, image_scale=image_scale, camera_index=i
            )
            img = camera_set.get_camera_by_index(i).get_image(image_scale=image_scale)
            rendered = rendered[..., : img.shape[-1]]
            composite = (
                np.clip(np.concatenate((img, rendered, (img + rendered) / 2.0)), 0, 1)
                * 255
            ).astype(np.uint8)
            skimage.io.imsave(f"{save_folder}/render_{i:03d}.png", composite)
