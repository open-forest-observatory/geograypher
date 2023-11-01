import typing
from pathlib import Path
from skimage.transform import resize

import geopandas as gpd
import matplotlib.pyplot as plt
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
from shapely import Point, Polygon
from tqdm import tqdm

from multiview_prediction_toolkit.cameras import (
    PhotogrammetryCamera,
    PhotogrammetryCameraSet,
)
from multiview_prediction_toolkit.config import (
    NULL_TEXTURE_FLOAT_VALUE,
    NULL_TEXTURE_INT_VALUE,
    PATH_TYPE,
    VERT_ID,
    VIS_FOLDER,
)
from multiview_prediction_toolkit.utils.indexing import (
    ensure_float_labels,
)
from multiview_prediction_toolkit.utils.parsing import parse_transform_metashape
import logging


class TexturedPhotogrammetryMesh:
    def __init__(
        self,
        mesh_filename: PATH_TYPE,
        downsample_target: float = 1.0,
        transform_filename: PATH_TYPE = None,
        texture: typing.Union[PATH_TYPE, np.ndarray, None] = None,
        texture_kwargs: dict = {},
        discrete_label: bool = True,
        require_transform: bool = False,
    ):
        """_summary_

        Args:
            mesh_filename (PATH_TYPE): Path to the mesh, in a format pyvista can read
            downsample_target (float, optional): Downsample to this fraction of vertices. Defaults to 1.0.
            texture (typing.Union[PATH_TYPE, np.ndarray, None]): Texture or path to one. See more details in `load_texture` documentation
            texture_kwargs
            discrete_label (bool, optional): Is the label quanity discrete or continous
        """
        self.mesh_filename = Path(mesh_filename)
        self.downsample_target = downsample_target
        self.discrete_label = discrete_label

        self.pyvista_mesh = None
        self.pytorch3d_mesh = None
        self.texture = None
        self.verts = None
        self.faces = None
        self.vertex_texture = None
        self.face_texture = None
        self.local_to_epgs_4978_transform = None
        self.label_names = None

        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")

        # Load the mesh with the pyvista loader
        logging.info("Loading mesh")
        self.load_mesh(downsample_target=downsample_target)
        # Load the transform
        logging.info("Loading transform to EPSG:4326")
        self.load_transform_to_epsg_4326(
            transform_filename, require_transform=require_transform
        )
        # Load the texture
        logging.info("Loading texture")
        self.load_texture(texture, texture_kwargs)

    # Setup methods

    def load_mesh(
        self,
        downsample_target: float = 1.0,
    ):
        """Load the pyvista mesh and create the pytorch3d texture

        Args:
            downsample_target (float, optional):
                What fraction of mesh vertices to downsample to. Defaults to 1.0, (does nothing).
        """
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
        self.verts = self.pyvista_mesh.points.copy()
        # See here for format: https://github.com/pyvista/pyvista-support/issues/96
        self.faces = self.pyvista_mesh.faces.reshape((-1, 4))[:, 1:4].copy()

    def load_transform_to_epsg_4326(
        self, transform_filename: PATH_TYPE, require_transform: bool = False
    ):
        """
        Load the 4x4 transform projects points from their local coordnate system into EPSG:4326,
        the earth-centered, earth-fixed coordinate frame. This can either be from a CSV file specifying
        it directly or extracted from a Metashape camera output

        Args
            transform_filename (PATH_TYPE):
            require_transform (bool): Does a local-to-global transform file need to be available"
        Raises:
            FileNotFoundError: Cannot find texture file
            ValueError: Transform file doesn't have 4x4 matrix
        """
        if transform_filename is None:
            if require_transform:
                raise ValueError("Transform is required but not provided")
            # If not required, do nothing. TODO consider adding a warning
            return

        elif Path(transform_filename).suffix == ".xml":
            self.local_to_epgs_4978_transform = parse_transform_metashape(
                transform_filename
            )
        elif Path(transform_filename).suffix == ".csv":
            self.local_to_epgs_4978_transform = np.loadtxt(
                transform_filename, delimiter=","
            )
            if self.local_to_epgs_4978_transform.shape != (4, 4):
                raise ValueError(
                    f"Transform should be (4,4) but is {self.local_to_epgs_4978_transform.shape}"
                )
        else:
            if require_transform:
                raise ValueError(
                    f"Transform could not be loaded from {transform_filename}"
                )
            # Not set
            return

    def standardize_texture(self, texture_array: np.ndarray):
        # TODO consider coercing into a numpy array

        # Check the dimensions
        if texture_array.ndim == 1:
            texture_array = np.expand_dims(texture_array, axis=1)
        elif texture_array.ndim != 2:
            raise ValueError(
                f"Input texture should have 1 or 2 dimensions but instead has {texture_array.ndim}"
            )
        return texture_array

    def get_texture(
        self,
        request_vertex_texture: typing.Union[bool, None] = None,
        try_verts_faces_conversion: bool = True,
    ):
        # If this is unset, try to infer it
        if request_vertex_texture is None:
            if (self.vertex_texture is None and self.face_texture is None) or (
                self.vertex_texture is not None and self.face_texture is not None
            ):
                raise ValueError(
                    "Abigious which texture is requested, set requeste_vertex_texture appropriately"
                )

            # Assume that the only one available is being requested
            request_vertex_texture = self.vertex_texture is None

        if request_vertex_texture:
            if self.vertex_texture is not None:
                return self.standardize_texture(self.vertex_texture)
            elif try_verts_faces_conversion:
                self.set_texture(self.face_to_vert_texture(self.face_texture))
                self.vertex_texture
            else:
                raise ValueError(
                    "Vertex texture not present and conversion was not requested"
                )

        else:
            if self.face_texture is not None:
                return self.standardize_texture(self.face_texture)
            elif try_verts_faces_conversion:
                self.set_texture(self.vert_to_face_texture(self.vertex_texture))
                return self.face_texture
            else:
                raise ValueError(
                    "Face texture not present and conversion was not requested"
                )

    def set_texture(
        self,
        texture_array: np.ndarray,
        is_vertex_texture: typing.Union[bool, None] = None,
    ):
        texture_array = self.standardize_texture(texture_array)

        # If it is not specified whether this is a vertex texture, attempt to infer it from the shape
        if is_vertex_texture is None:
            # Check that the number of matches face or verts
            n_values = texture_array.shape[0]
            n_faces = self.faces.shape[0]
            n_verts = self.verts.shape[0]

            if n_verts == n_faces:
                raise ValueError(
                    "Cannot infer whether texture should be applied to vertices of faces because the number is the same"
                )
            elif n_values == n_verts:
                is_vertex_texture = True
            elif n_values == n_faces:
                is_vertex_texture = False
            else:
                raise ValueError(
                    f"The number of elements in the texture ({n_values}) did not match the number of faces ({n_faces}) or vertices ({n_verts})"
                )

        # Set the appropriate texture
        if is_vertex_texture:
            self.vertex_texture = texture_array
        else:
            self.face_texture = texture_array

    def load_texture(
        self,
        texture: typing.Union[PATH_TYPE, np.ndarray, None],
        texture_kwargs: dict = {},
    ):
        """Sets either self.face_texture or self.vertex_texture to an (n_{faces, verts}, m channels) array. Note that the other
           one will be left as None

        Args:
            texture (typing.Union[PATH_TYPE, np.ndarray, None]): This is either a numpy array or a file to one of the following
                * A numpy array file in ".npy" format
                * A vector file readable by geopandas and a label(s) specifying which column to use.
                  This should be dataset of polygons/multipolygons. Ideally, there should be no overlap between
                  regions with different labels. These regions may be assigned based on the order of the rows.
                * A raster file readable by rasterio. We may want to support using a subset of bands
            texture_kwargs (dict, optional): Keywords specific to different types of textures. Defaults to {}.
        """
        if texture is None:
            # Try to load it from the mesh
            # Note that this requires us to have not decimated yet
            texture = self.pyvista_mesh.active_scalars

            if texture is not None:
                self.set_texture(texture)
            # Assume that no texture will be needed, consider printing a warning
            return
        elif not isinstance(texture, np.ndarray):
            successfully_loaded = False
            # Try loading different types of files
            try:
                texture = np.load(texture, allow_pickle=True)
                successfully_loaded = True
            except:
                pass

            if not successfully_loaded:
                try:
                    gdf = gpd.read_file(texture)
                    column_name = texture_kwargs.get("column_name")
                    self.get_values_for_verts_from_vector(
                        column_names=column_name,
                        geopandas_df=gdf,
                        set_vertex_texture=True,
                    )

                    successfully_loaded = True
                except (ValueError, AttributeError):
                    pass

            if not successfully_loaded:
                try:
                    # TODO
                    src = rio.open(texture)
                    breakpoint()
                except ValueError:
                    pass

            if not successfully_loaded:
                try:
                    # TODO
                    point_data = self.pyvista_mesh.point_data[texture]
                    breakpoint()
                except KeyError:
                    raise ValueError(f"Texture {texture} not found in the pyvista mesh")

            if not successfully_loaded:
                raise ValueError(f"Could not load texture for {texture}")

        else:
            self.set_texture(texture_array=texture)

    def get_label_names(self):
        return self.label_names

    def set_label_names(self, label_names):
        self.label_names = label_names

    def create_pytorch3d_mesh(
        self,
        vert_texture: np.ndarray = None,
        force_recreation: bool = False,
    ):
        """Create the pytorch_3d_mesh

        Args:
            vert_texture (np.ndarray, optional):
                Optional texture, (n_verts, n_channels). In the range [0, 1]. Defaults to None.
            force_recreation (bool, optional):
                if True, create a new mesh even if one already exists
        """
        # No-op if a mesh exists already
        if not force_recreation and self.pytorch3d_mesh is not None:
            return

        # Create the texture object if provided
        if vert_texture is not None:
            vert_texture = (
                torch.Tensor(vert_texture).to(torch.float).to(self.device).unsqueeze(0)
            )
            if len(vert_texture.shape) == 2:
                vert_texture = vert_texture.unsqueeze(-1)
            texture = TexturesVertex(verts_features=vert_texture).to(self.device)
        else:
            texture = None

        # Create the pytorch mesh
        self.pytorch3d_mesh = Meshes(
            verts=[torch.Tensor(self.verts).to(self.device)],
            faces=[torch.Tensor(self.faces).to(self.device)],
            textures=texture,
        ).to(self.device)

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

        # Check if it's lat, lon which is y-first
        is_lat_lon = int(crs == pyproj.CRS.from_epsg(4326))
        df = pd.DataFrame(
            {
                "east": verts_in_geopolygon_crs[:, 0 + is_lat_lon],
                "north": verts_in_geopolygon_crs[:, 1 - is_lat_lon],
            }
        )
        # Create a column of Point objects to use as the geometry
        df["geometry"] = gpd.points_from_xy(df["east"], df["north"])
        points = gpd.GeoDataFrame(df, crs=crs)

        # Add an index column because the normal index will not be preserved in future operations
        points[VERT_ID] = df.index

        return points

    # Transform labels face<->vertex methods

    def face_to_vert_texture(self, face_IDs):
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

    def vert_to_face_texture(self, vert_IDs):
        if vert_IDs is None:
            raise ValueError("None")

        # Each row contains the IDs of each vertex
        IDs_per_face = vert_IDs[self.faces]
        # Now we need to "vote" for the best one
        max_ID = int(np.nanmax(vert_IDs))
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
        # Set any faces with zero counts to the null value
        most_common_class_per_face[zeros_mask] = NULL_TEXTURE_FLOAT_VALUE

        return most_common_class_per_face

    # Operations on vector data
    def get_values_for_verts_from_vector(
        self,
        column_names: typing.List[str],
        vector_file: PATH_TYPE = None,
        geopandas_df: gpd.GeoDataFrame = None,
        set_vertex_texture: bool = False,
    ) -> np.ndarray:
        """Get the value from a dataframe for each vertex

        Args:
            column_names (str): Which column to obtain data from
            geopandas_df (GeoDataFrame, optional): Data to use.
            vector_file (PATH_TYPE, optional): Path to data that can be loaded by geopandas
            set_vertex_texture (bool, optional): If True and only one column name is provided, set self.vertex_texture to the result

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

        if len(geopandas_df) == 1:
            # TODO benchmark if this is faster
            points_in_polygons = verts_df.intersection(geopandas_df["geometry"][0])
        else:
            # Select points that are within the polygons
            points_in_polygons = gpd.tools.overlay(
                verts_df, geopandas_df, how="intersection"
            )

        # If it's one string, make it a one-length array
        if isinstance(column_names, str):
            column_names = [column_names]
        # Get the index array
        index_array = points_in_polygons[VERT_ID].to_numpy()

        output_dict = {}
        # Extract the data from each
        for column_name in column_names:
            # Create an array corresponding to all the points and initialize to NaN
            values = np.full(shape=verts_df.shape[0], fill_value=np.nan)
            # Assign points that are inside a given tree with that tree's ID
            values[index_array], self.label_names = ensure_float_labels(
                query_array=points_in_polygons[column_name],
                full_array=geopandas_df[column_name],
            )
            output_dict[column_name] = values
        # If only one name was requested, just return that
        if len(column_names) == 1:
            output_values = np.array(list(output_dict.values())[0])
            if set_vertex_texture:
                self.set_texture(output_values, is_vertex_texture=True)

            return output_values
        # Else return a dict of all requested values
        return output_dict

    def save_mesh(self, savepath: PATH_TYPE, save_vert_texture: bool = True):
        # TODO consider moving most of this functionality to a utils file
        if save_vert_texture:
            vert_texture = self.get_texture(request_vertex_texture=True)
            n_channels = vert_texture.shape[1]

            if n_channels == 1:
                vert_texture = np.repeat(vert_texture, (1, 3))
            if n_channels > 3:
                logging.warning(
                    "Too many channels to save, attempting to treat them as class probabilities and take the argmax"
                )
                # Take the argmax
                vert_texture = np.nanargmax(vert_texture, axis=1, keepdims=True)
                # Replace nan with 255
                vert_texture = np.nan_to_num(vert_texture, nan=NULL_TEXTURE_INT_VALUE)
                # Expand to the right number of channels
                vert_texture = np.repeat(vert_texture, repeats=(1, 3))

            vert_texture = vert_texture.astype(np.uint8)
        else:
            vert_texture = None

        self.pyvista_mesh.save(savepath, texture=vert_texture)

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
        # TODO consider if there's a case where this is wrong and the axes need to be switched
        x_points = verts_in_raster_CRS[:, 0].tolist()
        y_points = verts_in_raster_CRS[:, 1].tolist()

        # Zip them together
        zipped_locations = zip(x_points, y_points)
        sampling_iter = tqdm(
            zipped_locations,
            desc="Sampling values from raster",
            total=verts_in_raster_CRS.shape[0],
        )
        # Sample the raster file and squeeze if single channel
        sampled_raster_values = np.squeeze(np.array(list(raster.sample(sampling_iter))))

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

        fragments = rasterizer(self.pytorch3d_mesh)
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
        set_null_texture_to_value: float = None,
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
                self.face_texture is not None
                and self.face_texture.shape[0] == self.faces.shape[0]
            ):
                pass
            if (
                self.vertex_texture is not None
                and self.vertex_texture.shape[0] == self.verts.shape[0]
            ):
                self.face_texture = self.vert_to_face_texture(self.vertex_texture)
            else:
                raise ValueError("No texture for rendering")
            self.create_pytorch3d_mesh(self.vertex_texture)
        else:
            if self.pytorch3d_mesh.textures is None:
                self.create_pytorch3d_mesh(self.vertex_texture)

        # Get the photogrametery camera
        pg_camera = camera_set.get_camera_by_index(camera_index)

        # Compute the pixel-to-vertex correspondences, this is expensive
        p3d_camera, fragments = self.get_rasterization_results_pytorch3d(
            pg_camera, image_scale=image_scale
        )

        if shade_by_indexing:
            pix_to_face = fragments.pix_to_face[0, :, :, 0].cpu().numpy().flatten()
            pix_to_label = self.face_texture[pix_to_face]
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
            label_img = shader(fragments, self.pytorch3d_mesh)[0].cpu().numpy()

        # Remap the value for null pixels
        if set_null_texture_to_value is not None:
            label_img[label_img == NULL_TEXTURE_FLOAT_VALUE] = set_null_texture_to_value

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
        off_screen = (not interactive) or (screenshot_filename is not None)
        if off_screen:
            pv.start_xvfb()
        # Create the plotter which may be onscreen or off
        plotter = pv.Plotter(off_screen=off_screen)

        # If the vis scalars are None, use the vertex IDs
        if vis_scalars is None:
            vis_scalars = self.get_texture(
                # Request vertex texture if both are available
                request_vertex_texture=(
                    True
                    if (
                        self.vertex_texture is not None
                        and self.face_texture is not None
                    )
                    else None
                )
            ).astype(float)
            vis_scalars[vis_scalars < 0] = np.nan

        is_rgb = (
            self.pyvista_mesh.active_scalars_name == "RGB"
            if vis_scalars is None
            else (len(vis_scalars.shape[1]) > 1)
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
        render_image_scale=1.0,
        camera_indices=None,
        render_folder: PATH_TYPE = "renders",
        make_composites: bool = False,
        blend_composite: bool = True,
        save_native_resolution: bool = False,
        set_null_texture_to_value: float = 255,
    ):
        """Render an image from the viewpoint of each specified camera and save a composite

        Args:
            camera_set (PhotogrammetryCameraSet): Camera set to use for rendering
            render_image_scale (float, optional):
                Multiplier on the real image scale to obtain size for rendering. Lower values
                yield a lower-resolution render but the runtime is quiker. Defaults to 1.0.
            camera_indices (ArrayLike | NoneType, optional): Indices to render. If None, render all in a random order
            render_folder (PATH_TYPE, optional): Save images to this folder within vis. Default "renders"
            make_composites (bool, optional): Should a triple composite the original image be saved
            blend_composite (bool, optional): Should the real and rendered image be saved, rather than inserting a channel of the render into the real
        """
        # Render each image individually.
        # TODO this could be accelerated by inteligent batching
        if camera_indices is None:
            camera_indices = np.arange(camera_set.n_cameras())
            np.random.shuffle(camera_indices)

        save_folder = Path(VIS_FOLDER, render_folder)
        save_folder.mkdir(parents=True, exist_ok=True)
        logging.info(f"Saving renders to {save_folder}")

        for i in tqdm(camera_indices, desc="Saving renders"):
            rendered = self.render_pytorch3d(
                camera_set=camera_set,
                camera_index=i,
                image_scale=render_image_scale,
                set_null_texture_to_value=set_null_texture_to_value,
            )

            # Clip channels if needed
            if rendered.ndim == 3:
                rendered = rendered[..., :3]

            if save_native_resolution:
                native_size = camera_set.get_camera_by_index(i).get_image_size()
                # Upsample using nearest neighbor interpolation for discrete labels and
                # bilinear for non-discrete
                rendered = resize(
                    rendered, native_size, order=(0 if self.discrete_label else 1)
                )

            if make_composites:
                real_img = camera_set.get_camera_by_index(i).get_image(
                    image_scale=(1.0 if save_native_resolution else render_image_scale)
                )[..., :3]

                if blend_composite:
                    combined = (real_img + rendered) / 2.0
                else:
                    combined = real_img.copy().astype(float)
                    combined[..., 0] = rendered[..., 0]
                rendered = np.clip(
                    np.concatenate((real_img, rendered, combined), axis=1),
                    0,
                    1,
                )

            # rendered = (rendered * 255).astype(np.uint8)
            rendered = rendered.astype(np.uint8)
            output_filename = Path(
                save_folder, camera_set.get_image_filename(i, absolute=False)
            )
            # This may create nested folders in the output dir
            output_filename.mkdir(parents=True, exist_ok=True)
            output_filename = str(output_filename.with_suffix(".png"))
            # Save the image
            skimage.io.imsave(output_filename, rendered)
