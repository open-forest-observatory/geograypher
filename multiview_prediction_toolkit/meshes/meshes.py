from pathlib import Path

import numpy as np
import pyproj
import pyvista as pv
import skimage
import torch
import typing
import matplotlib.pyplot as plt
from pytorch3d.renderer import (
    MeshRasterizer,
    RasterizationSettings,
    TexturesVertex,
    AmbientLights,
    HardGouraudShader,
)
import rasterio as rio
from collections import Counter
from pytorch3d.structures import Meshes
from tqdm import tqdm

from multiview_prediction_toolkit.cameras import (
    PhotogrammetryCamera,
    PhotogrammetryCameraSet,
)
from shapely import Polygon, Point
import geopandas as gpd
import pandas as pd
from multiview_prediction_toolkit.config import PATH_TYPE, VIS_FOLDER


class TexturedPhotogrammetryMesh:
    def __init__(
        self,
        mesh_filename: PATH_TYPE,
        downsample_target: float = 1.0,
        texture: np.ndarray = None,
        use_pytorch3d_mesh: bool = True,
        discrete_label: bool = True,
    ):
        """_summary_

        Args:
            mesh_filename (PATH_TYPE): Path to the mesh, in a format pyvista can read
            downsample_target (float, optional): Downsample to this fraction of vertices. Defaults to 1.0.
            texture (np.ndarray): The texture to apply to the pytorch3d mesh
            use_pytorch3d_mesh (bool, optional): Set to False if unneeded or will be done later
            discrete_label (bool, optional): Is the label quanity discrete or continous
        """
        self.mesh_filename = Path(mesh_filename)
        self.downsample_target = downsample_target
        self.texture = texture
        self.use_pytorch3d_mesh = use_pytorch3d_mesh
        self.discrete_label = discrete_label

        self.pyvista_mesh = None
        self.pytorch_mesh = None
        self.verts = None
        self.faces = None
        self.vertex_IDs = None
        self.face_IDs = None
        self.local_to_epgs_4978_transform = None

        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")

        # Load the mesh with the pyvista loader
        self.load_mesh(downsample_target=downsample_target)
        # Create the pytorch3d mesh
        self.create_pytorch_3d_mesh(vert_texture=texture)

    # Setup methods

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
            print(
                f"Required transform file {transform_filename} file could not be found"
            )
            self.local_to_epgs_4978_transform = np.eye(4)

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

        self.verts = verts.copy()
        self.faces = faces.copy()

    def create_pytorch_3d_mesh(
        self,
        vert_texture: np.ndarray = None,
        force_creation: bool = None,
    ):
        """Create the pytorch_3d_mesh

        Args:
            vert_texture (np.ndarray, optional):
                Optional texture, (n_verts, n_channels). In the range [0, 1]. Defaults to None.
            force_creation (bool, optional): If None, rely on self.use_pytorch3d_mesh. Otherwise True creates the mesh and False doesn't

        """
        # No op
        if (not self.use_pytorch3d_mesh and force_creation is not True) or (
            force_creation is False
        ):
            return

        # Create the texture object if provided
        if vert_texture is not None:
            vert_texture = torch.Tensor(vert_texture).to(self.device).unsqueeze(0)
            if len(vert_texture.shape) == 2:
                vert_texture = vert_texture.unsqueeze(-1)
            texture = TexturesVertex(verts_features=vert_texture).to(self.device)
        else:
            texture = None

        # Create the pytorch mesh
        self.pytorch_mesh = Meshes(
            verts=[torch.Tensor(self.verts).to(self.device)],
            faces=[torch.Tensor(self.faces).to(self.device)],
            textures=texture,
        ).to(self.device)

    def texture_with_binary_mask(
        self,
        binary_mask: np.ndarray,
        color_true: list,
        color_false: list,
        vis: bool = False,
    ):
        """Color the pyvista and pytorch3d meshes based on a binary mask and two colors
        TODO Consider extending this to multiclass

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
        # Create pytorch3d mesh
        self.create_pytorch_3d_mesh(vert_texture=colors_tensor)

    # Vertex methods

    def transform_vertices(self, transform_4x4: np.ndarray, in_place: bool = False):
        """Apply a transform to the vertex coordinates

        Args:
            transform_4x4 (np.ndarray): Transform to be applied
            in_place (bool): Should the vertices be updated for all member objects
        """
        homogenous_local_points = np.vstack(
            (self.verts.T, np.ones(self.verts.shape[0]))
        )
        transformed_local_points = transform_4x4 @ homogenous_local_points
        transformed_local_points = transformed_local_points[:3].T

        # Overwrite existing vertices in both pytorch3d and pyvista mesh
        if in_place:
            self.verts = transformed_local_points.copy()
            self.pyvista_mesh.points = transformed_local_points.copy()
            self.create_pytorch_3d_mesh(vert_texture=self.texture)
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

        # TODO figure out why this conversion was required. I think it was some typing issue
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

    def get_verts_geodataframe(self, crs: pyproj.CRS) -> gpd.GeoDataFrame:
        """Obtain the vertices as a dataframe

        Args:
            crs (pyproj.CRS): The CRS to use

        Returns:
            gpd.GeoDataFrame: A dataframe with all the vertices
        """

        # Get the vertices in the same CRS as the geofile
        verts_in_geopolygon_crs = self.get_vertices_in_CRS(crs)

        # Taken from https://www.matecdev.com/posts/point-in-polygon.html
        # Convert points into georeferenced dataframe
        # TODO consider removing this line since it's not strictly needed
        df = pd.DataFrame(
            {
                "east": verts_in_geopolygon_crs[:, 0],
                "north": verts_in_geopolygon_crs[:, 1],
            }
        )
        # Create a column of Point objects to use as the geometry
        df["coords"] = [Point(xy) for xy in zip(df["east"], df["north"])]
        points = gpd.GeoDataFrame(df, geometry="coords", crs=crs)

        # Add an index column because the normal index will not be preserved in future operations
        points["id"] = df.index

        return points

    # Transform labels face<->vertex methods

    def face_to_vert_IDs(self, face_IDs):
        """_summary_

        Args:
            face_IDs (np.array): (n_faces,) The integer IDs of the faces
        """
        raise NotImplementedError()
        # TODO figure how to have a NaN class that
        for i in tqdm(range(self.verts.shape[0])):
            # Find which faces are using this vertex
            matching = np.sum(self.faces == i, axis=1)
            # matching_inds = np.where(matching)[0]
            # matching_IDs = face_IDs[matching_inds]
            # most_common_ind = Counter(matching_IDs).most_common(1)

    def vert_to_face_IDs(self, vert_IDs):
        # Each row contains the IDs of each vertex
        IDs_per_face = vert_IDs[self.faces]
        # Now we need to "vote" for the best one
        max_ID = np.max(vert_IDs)
        # TODO consider using unique if these indices are sparse
        counts_per_class_per_face = np.array(
            [np.sum(IDs_per_face == i, axis=1) for i in range(max_ID + 1)]
        ).T
        # Check which entires had no classes reported and mask them out
        # TODO consider removing these rows beforehand
        zeros_mask = np.all(counts_per_class_per_face == 0, axis=1)
        # We want to fairly tiebreak since np.argmax will always take th first index
        # This is hard to do in a vectorized way, so we just add a small random value
        # independently to each element
        counts_per_class_per_face = (
            counts_per_class_per_face
            + np.random.random(counts_per_class_per_face.shape) * 0.5
        )
        most_common_class_per_face = np.argmax(counts_per_class_per_face, axis=1)
        most_common_class_per_face[zeros_mask] = -1

        return most_common_class_per_face

    # Operations on vector data
    def get_values_for_verts_from_vector(
        self,
        column_names: typing.List[str],
        vector_file: PATH_TYPE = None,
        geopandas_df: gpd.GeoDataFrame = None,
    ) -> np.ndarray:
        """Get the value from a dataframe for each vertex

        Args:
            column_names (str): Which column to obtain data from
            geopandas_df (GeoDataFrame, optional): Data to use.
            vector_file (PATH_TYPE, optional): Path to data that can be loaded by geopandas

        Returns:
            np.ndarray: Array of values for each vertex if there is one column name or
            dict[np.ndarray]: A dict mapping from column names to numpy arrays
        """
        if vector_file is None and geopandas_df is None:
            raise ValueError("Must provide either vector_file or geopandas_df")

        if geopandas_df is None:
            geopandas_df = gpd.read_file(vector_file)

        # Get a dataframe of vertices
        verts_df = self.get_verts_geodataframe(geopandas_df.crs)

        # Select points that are within the polygons
        points_in_polygons = gpd.tools.overlay(
            verts_df, geopandas_df, how="intersection"
        )

        # If it's one string, make it a one-length array
        if isinstance(column_names, str):
            column_names = [column_names]

        # Get the index array
        index_array = points_in_polygons["id"].to_numpy()

        output_dict = {}
        # Extract the data from each
        for column_name in column_names:
            # Create an array corresponding to all the points and initialize to NaN
            values = np.full(shape=verts_df.shape[0], fill_value=np.nan)
            # Assign points that are inside a given tree with that tree's ID
            values[index_array] = points_in_polygons[column_name].to_numpy()
            output_dict[column_name] = values

        # If only one name was requested, just return that
        if len(column_names) == 1:
            return list(output_dict.values())[0]

        return output_dict

    def export_face_labels_vector(
        self,
        face_labels: np.ndarray,
        export_file: PATH_TYPE = None,
        export_crs: pyproj.CRS = pyproj.CRS.from_epsg(4326),
        label_names: typing.Tuple = None,
        drop_na: bool = True,
        vis: bool = True,
        vis_kwargs: typing.Dict = {},
    ) -> gpd.GeoDataFrame:
        """Export the labels for each face as a on-per-class multipolygon

        Args:
            face_labels (np.ndarray): Array of integer labels and potentially nan
            export_file (PATH_TYPE, optional):
                Where to export. The extension must be a filetype that geopandas can write.
                Defaults to None, if unset, nothing will be written.
            export_crs (pyproj.CRS, optional): What CRS to export in.. Defaults to pyproj.CRS.from_epsg(4326), lat lon.
            label_names (typing.Tuple, optional): Optional names, that are indexed by the labels. Defaults to None.
            drop_na (bool, optional): Should the faces with the nan class be discarded. Defaults to True.
            vis: should the result be visualzed
            vis_kwargs: keyword argmument dict for visualization

        Raises:
            ValueError: If the wrong number of faces labels are provided

        Returns:
            gpd.GeoDataFrame: Merged data
        """
        # Check that the correct number of labels are provided
        if len(face_labels) != self.faces.shape[0]:
            raise ValueError()

        # Get the mesh vertices in the desired export CRS
        verts_in_crs = self.get_vertices_in_CRS(export_crs)
        # Get a triangle in geospatial coords for each face
        # Only report the x, y values and not z
        face_polygons = [
            Polygon(verts_in_crs[face_IDs][:, :2]) for face_IDs in self.faces
        ]
        # Create a geodata frame from these polygons
        individual_polygons_df = gpd.GeoDataFrame(
            {"labels": face_labels}, geometry=face_polygons, crs=export_crs
        )
        # Merge these triangles into a multipolygon for each class
        # This is the expensive step
        aggregated_df = individual_polygons_df.dissolve(
            by="labels", as_index=False, dropna=drop_na
        )

        # Add names if present
        if label_names is not None:
            names = [
                (label_names[int(label)] if label is not np.nan else np.nan)
                for label in aggregated_df["labels"].tolist()
            ]
            aggregated_df["names"] = names

        # Export if a file is provided
        if export_file is not None:
            aggregated_df.to_file(export_file)

        # Vis if requested
        if vis:
            aggregated_df.plot(
                column="names" if label_names is not None else "labels",
                aspect=1,
                legend=True,
                **vis_kwargs,
            )
            plt.show()

        return aggregated_df

    # Operations on raster files

    def get_vert_values_from_raster_file(
        self, raster_file: PATH_TYPE, return_verts_in_CRS: bool = False
    ):
        """Compute the height above groun for each point on the mesh

        Args:
            raster_file (PATH_TYPE, optional): The path to the geospatial raster file.
            return_verts_in_CRS (bool, optional): Return the vertices transformed into the raster CRS

        Returns:
            np.ndarray: samples from raster. Either (n_verts,) or (n_verts, n_raster_channels)
            np.ndarray (optional): (n_verts, 3) the vertices in the raster CRS
        """
        # Open the DEM file
        raster = rio.open(raster_file)
        # Get the mesh points in the coordinate reference system of the DEM
        verts_in_raster_CRS = self.get_vertices_in_CRS(raster.crs)

        # Get the points as a list
        x_points = verts_in_raster_CRS[:, 1].tolist()
        y_points = verts_in_raster_CRS[:, 0].tolist()

        # Zip them together
        zipped_locations = zip(x_points, y_points)
        # Sample the raster file and squeeze if single channel
        sampled_raster_values = np.squeeze(
            np.array(list(raster.sample(zipped_locations)))
        )

        if return_verts_in_CRS:
            return sampled_raster_values, verts_in_raster_CRS

        return sampled_raster_values

    def get_height_above_ground(
        self, DEM_file: PATH_TYPE, threshold: float = None
    ) -> np.ndarray:
        """Return height above ground for a points in the mesh and a given DEM

        Args:
            DEM_file (PATH_TYPE): Path to the DEM raster
            threshold (float, optional):
                If not None, return a boolean mask for points under this height. Defaults to None.

        Returns:
            np.ndarray: Either the height above ground or a boolean mask for ground points
        """
        # Get the height from the DEM and the points in the same CRS
        DEM_heights, verts_in_raster_CRS = self.get_vert_values_from_raster_file(
            DEM_file, return_verts_in_CRS=True
        )
        # Subtract the two to get the height above ground
        height_above_ground = verts_in_raster_CRS[:, 2] - DEM_heights

        # If the threshold is not None, return a boolean mask that is true for ground points
        if threshold is not None:
            # Return boolean mask
            return height_above_ground < threshold
        # Return height above ground
        return height_above_ground

    # Expensive pixel-to-vertex operations

    def get_rasterization_results_pytorch3d(
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
            _, fragments = self.get_rasterization_results_pytorch3d(
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

    def render_pytorch3d(
        self,
        camera_set: PhotogrammetryCameraSet,
        camera_index: int,
        image_scale: float = 1.0,
        shade_by_indexing: bool = None,
    ):
        """Render an image from the viewpoint of a single camera
        # TODO include an option to specify whether indexing or shading is used

        Args:
            camera_set (PhotogrammetryCameraSet): Camera set to use for rendering
            camera_index (int): which camera to render
            image_scale (float, optional):
                Multiplier on the real image scale to obtain size for rendering. Lower values
                yield a lower-resolution render but the runtime is quiker. Defaults to 1.0.
            shade_by_indexing (bool, optional): Use indexing rather than a pytorch3d shader. Useful for integer labels
        """
        if shade_by_indexing is None:
            shade_by_indexing = self.discrete_label

        # Check to make sure required data is available
        if shade_by_indexing:
            if (
                self.face_IDs is not None
                and self.face_IDs.shape[0] == self.faces.shape[0]
            ):
                pass
            if (
                self.vertex_IDs is not None
                and self.vertex_IDs.shape[0] == self.verts.shape[0]
            ):
                self.face_IDs = self.vert_to_face_IDs(self.vertex_IDs)
            else:
                raise ValueError("No texture for rendering")
        else:
            if self.pytorch_mesh.textures is None:
                self.create_pytorch_3d_mesh(self.vertex_IDs)

        breakpoint()
        # Get the photogrametery camera
        pg_camera = camera_set.get_camera_by_index(camera_index)

        # Compute the pixel-to-vertex correspondences, this is expensive
        p3d_camera, fragments = self.get_rasterization_results_pytorch3d(
            pg_camera, image_scale=image_scale
        )

        if shade_by_indexing:
            pix_to_face = fragments.pix_to_face[0, :, :, 0].cpu().numpy().flatten()
            pix_to_label = self.face_IDs[pix_to_face]
            img_size = pg_camera.get_image_size(image_scale=image_scale)
            label_img = np.reshape(pix_to_label, img_size)
        else:
            # Create ambient light so it doesn't effect the color
            lights = AmbientLights(device=self.device)
            # Create a shader
            shader = HardGouraudShader(
                device=self.device, cameras=p3d_camera, lights=lights
            )

            # Render te images using the shader
            label_img = shader(fragments, self.pytorch_mesh)[0]

        return label_img

    # Visualization and saving methods
    def vis(
        self,
        interactive=True,
        camera_set: PhotogrammetryCameraSet = None,
        screenshot_filename: PATH_TYPE = None,
        vis_scalars=None,
        mesh_kwargs: typing.Dict = {},
        plotter_kwargs: typing.Dict = {},
    ):
        """Show the mesh and cameras

        Args:
            off_screen (bool, optional): Show offscreen
            camera_set (PhotogrammetryCameraSet, optional): Cameras to visualize. Defaults to None.
            screenshot_filename (PATH_TYPE, optional): Filepath to save to, will show interactively if None. Defaults to None.
            vis_scalars: Scalars to show
            mesh_kwargs: dict of keyword arguments for the mesh
            plotter_kwargs: dict of keyword arguments for the plotter
        """
        # Create the plotter which may be onscreen or off
        plotter = pv.Plotter(
            off_screen=(not interactive) or (screenshot_filename is not None)
        )

        # If the vis scalars are None, use the vertex IDs
        if vis_scalars is None and self.vertex_IDs is not None:
            vis_scalars = self.vertex_IDs.copy().astype(float)
            vis_scalars[vis_scalars < 0] = np.nan
        elif vis_scalars is None and self.face_IDs is not None:
            vis_scalars = self.face_IDs.copy().astype(float)
            vis_scalars[vis_scalars < 0] = np.nan

        is_rgb = (
            self.pyvista_mesh.active_scalars_name == "RGB"
            if vis_scalars is None
            else (len(vis_scalars.shape) > 1)
        )

        # Add the mesh
        plotter.add_mesh(
            self.pyvista_mesh,
            scalars=vis_scalars,
            rgb=is_rgb,
            **mesh_kwargs,
        )
        # If the camera set is provided, show this too
        if camera_set is not None:
            camera_set.vis(plotter, add_orientation_cube=True)
        # Show
        plotter.show(screenshot=screenshot_filename, **plotter_kwargs)

    def save_renders_pytorch3d(
        self,
        camera_set: PhotogrammetryCameraSet,
        image_scale=1.0,
        camera_indices=None,
        render_folder: PATH_TYPE = "renders",
        make_composites: bool = False,
    ):
        """Render an image from the viewpoint of each specified camera and save a composite

        Args:
            camera_set (PhotogrammetryCameraSet): Camera set to use for rendering
            image_scale (float, optional):
                Multiplier on the real image scale to obtain size for rendering. Lower values
                yield a lower-resolution render but the runtime is quiker. Defaults to 1.0.
            camera_indices (ArrayLike | NoneType, optional): Indices to render. If None, render all in a random order
            render_folder (PATH_TYPE, optional): Save images to this folder within vis. Default "renders"
            make_composites (bool, optional): Should a triple composite the original image be saved
        """
        # Render each image individually.
        # TODO this could be accelerated by inteligent batching
        if camera_indices is None:
            camera_indices = np.arange(camera_set.n_cameras())
            np.random.shuffle(camera_indices)

        save_folder = Path(VIS_FOLDER, render_folder)
        save_folder.mkdir(parents=True, exist_ok=True)

        for i in tqdm(camera_indices):
            rendered = self.render_pytorch3d(
                camera_set=camera_set,
                camera_index=i,
                image_scale=image_scale,
            )
            if make_composites:
                real_img = camera_set.get_camera_by_index(i).get_image(
                    image_scale=image_scale
                )

                # Repeat channel if it's a single channel
                if rendered.shape[-1] == 1:
                    rendered = np.tile(rendered, (1, 1, real_img.shape[-1]))

                rendered = (
                    np.clip(
                        np.concatenate(
                            (real_img, rendered, (real_img + rendered) / 2.0)
                        ),
                        0,
                        1,
                    )
                    * 255
                ).astype(np.uint8)

            # Save the image
            skimage.io.imsave(f"{save_folder}/render_{i:03d}.png", rendered)
