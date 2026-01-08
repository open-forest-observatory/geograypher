import hashlib
import json
import logging
import sys
import typing
from collections import defaultdict
from pathlib import Path
from time import time

import fiona
import geopandas as gpd
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyproj
import pyvista as pv
import rasterio as rio
import shapely
import skimage
import ubelt as ub
from scipy.spatial import KDTree
from shapely import MultiPolygon, Polygon
from skimage.transform import resize
from tqdm import tqdm

from geograypher.cameras import PhotogrammetryCamera, PhotogrammetryCameraSet
from geograypher.constants import (
    CACHE_FOLDER,
    CLASS_ID_KEY,
    CLASS_NAMES_KEY,
    EARTH_CENTERED_EARTH_FIXED_CRS,
    LAT_LON_CRS,
    NULL_TEXTURE_INT_VALUE,
    PATH_TYPE,
    RATIO_3D_2D_KEY,
    VERT_ID,
    VIS_FOLDER,
)
from geograypher.utils.files import ensure_containing_folder, ensure_folder
from geograypher.utils.geometric import batched_unary_union
from geograypher.utils.geospatial import (
    coerce_to_geoframe,
    ensure_non_overlapping_polygons,
    ensure_projected_CRS,
    get_projected_CRS,
)
from geograypher.utils.indexing import ensure_float_labels
from geograypher.utils.numeric import compute_3D_triangle_area, fair_mode_non_nan
from geograypher.utils.visualization import create_composite, create_pv_plotter


class TexturedPhotogrammetryMesh:
    def __init__(
        self,
        mesh: typing.Union[PATH_TYPE, pv.PolyData],
        input_CRS: pyproj.CRS,
        downsample_target: float = 1.0,
        texture: typing.Union[PATH_TYPE, np.ndarray, None] = None,
        texture_column_name: typing.Union[PATH_TYPE, None] = None,
        IDs_to_labels: typing.Union[PATH_TYPE, dict, None] = None,
        shift: typing.Union[np.ndarray, None] = None,
        ROI: typing.Union[
            gpd.GeoDataFrame, Polygon, MultiPolygon, PATH_TYPE, None
        ] = None,
        ROI_buffer_meters: float = 0,
        log_level: str = "INFO",
    ):
        """
        An object that represents a geospatial mesh with associated textures and supports various
        rendering options.


        Args:
            mesh (typing.Union[PATH_TYPE, pv.PolyData]):
                Path to the mesh in a filetype readable by pyvista or a pyvista mesh object.
            input_CRS (pyproj.CRS):
                The vertex coordinates of the input mesh should be interpreteted in this coordinate
                references system to georeference them.
            downsample_target (float, optional):
                Downsample so this fraction of vertices remain. Defaults to 1.0.
            texture (typing.Union[PATH_TYPE, np.ndarray, None], optional):
                Texture or path to one. See more details in `load_texture` documentation. Defaults
                to None.
            texture_column_name (typing.Union[PATH_TYPE, None], optional):
                The column to use as the label for a vector data input. Passed to `load_texture`.
                Defaults to None.
            IDs_to_labels (typing.Union[PATH_TYPE, dict, None], optional):
                dictionary or path to JSON file containing the mapping from integer IDs to string
                class names. Defaults to None.
            shift (typing.Union[np.ndarray, None], optional):
                Represents an [x, y, z] shift as a (3,) array. If provided, shift all vertex
                coordinates by this amount in the input_CRS frame. Defaults to None.
            ROI (typing.Union[ gpd.GeoDataFrame, Polygon, MultiPolygon, PATH_TYPE, None ], optional):
                Crop the mesh to this region. For more information see `select_mesh_ROI`. Defaults
                to None.
            ROI_buffer_meters (float, optional):
                Buffer the cropped region by this distance. For more information see
                `select_mesh_ROI`. Defaults to 0.
            log_level (str, optional):
                Controls what severity of messages are logged. Defaults to "INFO".
        """
        self.downsample_target = downsample_target

        self.pyvista_mesh = None
        self.texture = None
        self.vertex_texture = None
        self.face_texture = None
        self.IDs_to_labels = None
        # Create the plotter that will later be used to compute correspondences between pixels
        # and the mesh. Note that this is only done to prevent a memory leak from creating multiple
        # plotters. See https://github.com/pyvista/pyvista/issues/2252
        self.pix2face_plotter = create_pv_plotter(off_screen=True)
        self.face_polygons_cache = {}
        self.face_2d_3d_ratios_cache = {}

        self.logger = logging.getLogger(f"mesh_{id(self)}")
        self.logger.setLevel(log_level)
        # Potentially necessary for Jupyter
        # https://stackoverflow.com/questions/35936086/jupyter-notebook-does-not-print-logs-to-the-output-cell
        # If you don't check that there's already a handler, you can have situations with duplicated
        # print outs if you have multiple mesh objects
        if not self.logger.hasHandlers():
            self.logger.addHandler(logging.StreamHandler(stream=sys.stdout))

        # Load the mesh with the pyvista loader
        self.logger.info("Loading mesh")
        self.load_mesh(
            mesh=mesh,
            input_CRS=input_CRS,
            downsample_target=downsample_target,
            shift=shift,
            ROI=ROI,
            ROI_buffer_meters=ROI_buffer_meters,
        )
        # Load the texture
        self.logger.info("Loading texture")
        # load IDs_to_labels
        # if IDs_to_labels not provided, check the directory of the mesh and get the file if found
        if IDs_to_labels is None and isinstance(mesh, PATH_TYPE.__args__):
            possible_json = Path(Path(mesh).stem + "_IDs_to_labels.json")
            if possible_json.exists():
                IDs_to_labels = possible_json
        # convert IDs_to_labels from file to dict
        if isinstance(IDs_to_labels, PATH_TYPE.__args__):
            with open(IDs_to_labels, "r") as file:
                IDs_to_labels = json.load(file)
                IDs_to_labels = {int(id): label for id, label in IDs_to_labels.items()}
        self.load_texture(texture, texture_column_name, IDs_to_labels=IDs_to_labels)

    # Setup methods
    def load_mesh(
        self,
        mesh: typing.Union[PATH_TYPE, pv.PolyData],
        input_CRS: pyproj.CRS,
        downsample_target: float = 1.0,
        shift: typing.Union[np.ndarray, None] = None,
        ROI=None,
        ROI_buffer_meters=0,
        ROI_simplify_tol_meters=2,
    ):
        """Load the pyvista mesh and create the texture

        Args:
            mesh (typing.Union[PATH_TYPE, pv.PolyData]):
                Path to the mesh or actual mesh
            downsample_target (float, optional):
                What fraction of mesh vertices to downsample to. Defaults to 1.0, (does nothing).
            shift (typing.Union[np.ndarray, None], optional):
                Represents an [x, y, z] shift as a (3,) array. If provided, shift all vertex
                coordinates by this amount in the input_CRS frame. Defaults to None.
            ROI:
                See select_mesh_ROI. Defaults to None
            ROI_buffer_meters:
                See select_mesh_ROI. Defaults to 0.
            ROI_simplify_tol_meters:
                See select_mesh_ROI. Defaults to 2.
        """
        self.CRS = input_CRS

        if isinstance(mesh, pv.PolyData):
            # If a mesh is provided directly, copy it so input mesh isn't modified
            self.pyvista_mesh = mesh.copy()
        else:
            # Load the mesh using pyvista
            # TODO see if pytorch3d has faster/more flexible readers. I'd assume no, but it's good to check
            self.logger.info("Reading the mesh")
            self.pyvista_mesh = pv.read(mesh)

        # Up-cast to avoid quantization errors after we shift or transform to larger values
        self.pyvista_mesh.points = self.pyvista_mesh.points.astype(float)

        # If a shift is provided, shift all mesh vertices by this amount
        if shift is not None:
            self.pyvista_mesh.points += shift

        self.logger.info("Selecting an ROI from mesh")
        # Select a region of interest if needed
        self.pyvista_mesh = self.select_mesh_ROI(
            region_of_interest=ROI,
            buffer_meters=ROI_buffer_meters,
            simplify_tol_meters=ROI_simplify_tol_meters,
        )

        # Reproject to a meters-based CRS. TODO consider if there's a better option than ECEF.
        self.reproject_CRS(target_CRS=EARTH_CENTERED_EARTH_FIXED_CRS, inplace=True)

        # Downsample mesh and transfer active scalars from original mesh to downsampled mesh
        if downsample_target != 1.0:
            # TODO try decimate_pro and compare quality and runtime
            # TODO also see this decimation algorithm: https://pyvista.github.io/fast-simplification/
            self.logger.info("Downsampling the mesh")
            # Have a temporary mesh so we can use the original mesh to transfer the active scalars to the downsampled one
            downsampled_mesh_without_textures = self.pyvista_mesh.decimate(
                target_reduction=(1 - downsample_target)
            )
            self.logger.info(
                f"Requested downsampling {downsample_target}, actual downsampling {downsampled_mesh_without_textures.n_points / self.pyvista_mesh.n_points}"
            )
            self.pyvista_mesh = self.transfer_texture(downsampled_mesh_without_textures)
        self.logger.info("Extracting faces from mesh")
        # See here for format: https://github.com/pyvista/pyvista-support/issues/96
        self.faces = self.pyvista_mesh.faces.reshape((-1, 4))[:, 1:4].copy()

    def reproject_CRS(
        self, target_CRS: pyproj.CRS, inplace: bool = True
    ) -> typing.Optional[pv.PolyData]:
        """
        Convert the mesh into a new coordinate reference system. This is done by updating the
        location of each vertex using the mappings between the current coordinate reference system
        and the requested one, as implemented in pyproj.

        Note that if the CRS of the mesh is None, this operation will do nothing and the original
        vertex values will be returned un-transformed.

        Args:
            target_CRS (pyproj.CRS): The coordinate reference system to transform the mesh to.
            inplace (bool, optional): Should the self.pyvista_mesh and self.CRS attributes be
            updated. Otherwise, an updated copy of the mesh is returned and the original is left
            unchanged. Defaults to True.

        Returns:
            (pv.PolyData, optional): If `inplace==False`, a transformed pyvista mesh will be returned
        """
        # Check if the mesh has a valid CRS
        if self.CRS is None:
            self.logger.warning("mesh CRS is None, reproject_CRS is doing nothing")
            # If not, just return the original coordinates as if they had been transformed
            verts_in_output_CRS = np.array(self.pyvista_mesh.points)
        else:
            # Build a pyproj transfrormer from the current to the desired CRS
            transformer = pyproj.Transformer.from_crs(self.CRS, target_CRS)

            # Convert the mesh vertices to a numpy array
            mesh_verts = np.array(self.pyvista_mesh.points)

            # Transform the coordinates
            verts_in_output_CRS = transformer.transform(
                xx=mesh_verts[:, 0],
                yy=mesh_verts[:, 1],
                zz=mesh_verts[:, 2],
            )
            # Stack and transpose
            verts_in_output_CRS = np.vstack(verts_in_output_CRS).T

            # TODO figure out how to deal with the fact that this may no longer be a right-handed coordinate system
            # See comment in `get_vertices_in_CRS`

        if inplace:
            # Update the CRS
            self.CRS = target_CRS
            # Update the mesh points
            self.pyvista_mesh.points = pv.pyvista_ndarray(verts_in_output_CRS)
        else:
            # Create a copy of the mesh
            copied_mesh = self.pyvista_mesh.copy(deep=True)
            # Update its points
            copied_mesh.points = pv.pyvista_ndarray(verts_in_output_CRS)
            # Return the updated copy
            return copied_mesh

    def transfer_texture(self, downsampled_mesh):
        """Transfer texture from original mesh to a downsampled version using KDTree for nearest neighbor point searches

        Args:
            downsampled_mesh (pv.PolyData): The downsampled version of the original mesh

        Returns:
            pv.PolyData: The downsampled mesh with the transferred textures
        """
        # Only transfer textures if there are point based scalars in the original mesh
        if self.pyvista_mesh.point_data:
            # Store original mesh points in KDTree for nearest neighbor search
            kdtree = KDTree(self.pyvista_mesh.points)

            # For ecah point in the downsampled mesh find the nearest neighbor point in the original mesh
            _, nearest_neighbor_indices = kdtree.query(downsampled_mesh.points)

            # Iterate over all the point based scalars
            for scalar_name in self.pyvista_mesh.point_data.keys():
                # Retrieve scalar data of appropriate index using the nearest neighbor indices
                transferred_scalars = self.pyvista_mesh.point_data[scalar_name][
                    nearest_neighbor_indices
                ]
                # Set the corresponding scalar data in the downsampled mesh
                downsampled_mesh.point_data[scalar_name] = transferred_scalars

            # Set active mesh of downsampled mesh
            if self.pyvista_mesh.active_scalars_name:
                downsampled_mesh.active_scalars_name = (
                    self.pyvista_mesh.active_scalars_name
                )
        else:
            self.logger.warning(
                "Textures not transferred, active scalars data is assoicated with cell data not point data"
            )
        return downsampled_mesh

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
        if self.vertex_texture is None and self.face_texture is None:
            return

        # If this is unset, try to infer it
        if request_vertex_texture is None:
            if self.vertex_texture is not None and self.face_texture is not None:
                raise ValueError(
                    "Ambigious which texture is requested, set request_vertex_texture appropriately"
                )

            # Assume that the only one available is being requested
            request_vertex_texture = self.vertex_texture is not None

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
                face_texture = self.vert_to_face_texture(
                    self.vertex_texture, discrete=self.is_discrete_texture()
                )
                self.set_texture(face_texture)
                return self.face_texture
            else:
                raise ValueError(
                    "Face texture not present and conversion was not requested"
                )

    def is_discrete_texture(self):
        return self.IDs_to_labels is not None

    def set_texture(
        self,
        texture_array: np.ndarray,
        IDs_to_labels: typing.Union[None, dict] = None,
        all_discrete_texture_values: typing.Union[typing.List, None] = None,
        is_vertex_texture: typing.Union[bool, None] = None,
        delete_existing: bool = True,
        update_IDs_to_labels: bool = True,
    ):
        """Set the internal texture representation

        Args:
            texture_array (np.ndarray):
                The array of texture values. The first dimension must be the length of faces or
                verts. A second dimension is optional.
            IDs_to_labels (typing.Union[None, dict], optional):
                Mapping from integer IDs to string names. Defaults to None.
            all_discrete_texture_values (typing.Union[typing.List, None], optional):
                Are all the texture values known to be discrete, representing IDs. Computed from
                the data if not set. Defaults to None.
            is_vertex_texture (typing.Union[bool, None], optional):
                Are the texture values supposed to correspond to the vertices. Computed from the
                data if not set. Defaults to None.
            delete_existing (bool, optional):
                Delete the existing texture when the other one (face, vertex) is set. Defaults to True.
            update_IDs_to_labels (bool, optional):
                Should IDs to labels be updated based on either the provided IDs_to_labels or the
                derived ones. Defaults to True.

        Raises:
            ValueError: If the size of the texture doesn't match the number of either faces or vertices
            ValueError: If the number of faces and vertices are the same and is_vertex_texture isn't set
        """
        # Ensure that the texture is 2D and a numpy array
        texture_array = self.standardize_texture(texture_array)

        if texture_array.ndim == 2 and texture_array.shape[1] != 1:
            # If it is more than one column, it's assumed to be a real-valued
            # quantity and we try to cast it to a float
            texture_array = texture_array.astype(float)
            self.IDs_to_labels = None
        else:
            if IDs_to_labels is None:
                texture_array, derived_IDs_to_labels = ensure_float_labels(
                    texture_array, full_array=all_discrete_texture_values
                )
                # If requested, record these new IDs_to_labels
                if update_IDs_to_labels:
                    self.IDs_to_labels = derived_IDs_to_labels
            else:
                # Create the inverse mapping, returning nan for anything not in it
                labels_to_IDs = defaultdict(lambda: np.nan)
                labels_to_IDs.update({v: k for k, v in IDs_to_labels.items()})

                # Ensure the mapping is 1-to-1, that there are no collisions in the mapping
                if len(labels_to_IDs) != len(IDs_to_labels):
                    raise ValueError("IDs_to_labels is not a one-to-one mapping")

                # Check that the mapping only produces ints
                if not np.all([isinstance(v, int) for v in labels_to_IDs.values()]):
                    raise ValueError(
                        "The labels to IDs mapping does not produce only floats"
                    )

                # Perform the mapping
                texture_array = np.array(
                    [labels_to_IDs[l] for l in texture_array.squeeze()]
                )
                # Reinstate the squeezed dimension
                texture_array = np.expand_dims(texture_array, axis=1)

                # If requested, record these IDs to labels
                if update_IDs_to_labels:
                    self.IDs_to_labels = IDs_to_labels

        # If it is not specified whether this is a vertex texture, attempt to infer it from the shape
        # TODO consider refactoring to check whether it matches the number of one of them,
        # no matter whether is_vertex_texture is specified
        if is_vertex_texture is None:
            # Check that the number of matches face or verts
            n_values = texture_array.shape[0]
            n_faces = self.faces.shape[0]
            n_verts = self.pyvista_mesh.points.shape[0]

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

        # Set the appropriate texture and optionally delete the other one
        if is_vertex_texture:
            self.vertex_texture = texture_array
            if delete_existing:
                self.face_texture = None
        else:
            self.face_texture = texture_array
            if delete_existing:
                self.vertex_texture = None

    def load_texture(
        self,
        texture: typing.Union[str, PATH_TYPE, np.ndarray, None],
        texture_column_name: typing.Union[None, PATH_TYPE] = None,
        IDs_to_labels: typing.Union[PATH_TYPE, dict, None] = None,
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
            texture_column_name: The column to use as the label for a vector data input
            IDs_to_labels (typing.Union[None, dict]): Dictionary mapping from integer IDs to string class names
        """
        # The easy case, a texture is passed in directly
        if isinstance(texture, np.ndarray):
            self.set_texture(
                texture_array=texture,
                IDs_to_labels=IDs_to_labels,
            )
        # If the texture is None, try to load it from the mesh
        # Note that this requires us to have not decimated yet
        elif texture is None:
            # See if the mesh has a texture, else this will be None
            texture_array = self.pyvista_mesh.active_scalars

            if texture_array is not None:
                # Check if this was a really one channel that had to be tiled to
                # three for saving
                if len(texture_array.shape) == 2:
                    min_val_per_row = np.min(texture_array, axis=1)
                    max_val_per_row = np.max(texture_array, axis=1)
                    if np.array_equal(min_val_per_row, max_val_per_row):
                        # This is supposted to be one channel
                        texture_array = texture_array[:, 0].astype(float)
                        # Set any values that are the ignore int value to nan
                texture_array = texture_array.astype(float)
                texture_array[texture_array == NULL_TEXTURE_INT_VALUE] = np.nan

                self.set_texture(
                    texture_array,
                    IDs_to_labels=IDs_to_labels,
                )
            else:
                if IDs_to_labels is not None:
                    self.IDs_to_labels = IDs_to_labels
                # Assume that no texture will be needed, consider printing a warning
                self.logger.warn("No texture provided")
        else:
            # Try handling all the other supported filetypes
            texture_array = None
            all_values = None

            # Name of scalar in the mesh
            try:
                self.logger.warn(
                    "Trying to read texture as a scalar from the pyvista mesh:"
                )
                texture_array = self.pyvista_mesh[str(texture)]
                self.logger.warn("- success")
            except (KeyError, ValueError):
                self.logger.warn("- failed")

            # Numpy file
            if texture_array is None:
                try:
                    self.logger.warn("Trying to read texture as a numpy file:")
                    texture_array = np.load(texture, allow_pickle=True)
                    self.logger.warn("- success")
                except:
                    self.logger.warn("- failed")

            # Vector file
            if texture_array is None:
                try:
                    self.logger.warn("Trying to read texture as vector file:")
                    # TODO IDs to labels should be used here if set so the computed IDs are aligned with that mapping
                    texture_array, all_values = self.get_values_for_verts_from_vector(
                        column_names=texture_column_name,
                        vector_source=texture,
                    )
                    self.logger.warn("- success")
                except (IndexError, fiona.errors.DriverError):
                    self.logger.warn("- failed")

            # Raster file
            if texture_array is None:
                try:
                    # TODO
                    self.logger.warn("Trying to read as texture as raster file: ")
                    texture_array = self.get_vert_values_from_raster_file(texture)
                    self.logger.warn("- success")
                except:
                    self.logger.warn("- failed")

            # Error out if not set, since we assume the intent was to have a texture at this point
            if texture_array is None:
                raise ValueError(f"Could not load texture for {texture}")

            # This will error if something is wrong with the texture that was loaded
            self.set_texture(
                texture_array,
                all_discrete_texture_values=all_values,
                IDs_to_labels=IDs_to_labels,
            )

    def select_mesh_ROI(
        self,
        region_of_interest: typing.Union[
            gpd.GeoDataFrame, Polygon, MultiPolygon, PATH_TYPE, None
        ],
        buffer_meters: float = 0,
        simplify_tol_meters: int = 0,
        default_CRS: pyproj.CRS = pyproj.CRS.from_epsg(4326),
        return_original_IDs: bool = False,
    ):
        """Get a subset of the mesh based on geospatial data

        Args:
            region_of_interest (typing.Union[gpd.GeoDataFrame, Polygon, MultiPolygon, PATH_TYPE]):
                Region of interest. Can be a
                * dataframe, where all columns will be colapsed
                * A shapely polygon/multipolygon
                * A file that can be loaded by geopandas
            buffer_meters (float, optional): Expand the geometry by this amount of meters. Defaults to 0.
            simplify_tol_meters (float, optional): Simplify the geometry using this as the tolerance. Defaults to 0.
            default_CRS (pyproj.CRS, optional): The CRS to use if one isn't provided. Defaults to pyproj.CRS.from_epsg(4326).
            return_original_IDs (bool, optional): Return the indices into the original mesh. Defaults to False.

        Returns:
            pyvista.PolyData: The subset of the mesh
            np.ndarray: The indices of the points in the original mesh (only if return_original_IDs set)
            np.ndarray: The indices of the faces in the original mesh (only if return_original_IDs set)
        """
        if region_of_interest is None:
            return self.pyvista_mesh

        # Get the ROI into a geopandas GeoDataFrame
        self.logger.info("Standardizing ROI")
        if isinstance(region_of_interest, gpd.GeoDataFrame):
            ROI_gpd = region_of_interest
        elif isinstance(region_of_interest, (Polygon, MultiPolygon)):
            ROI_gpd = gpd.DataFrame(crs=default_CRS, geometry=[region_of_interest])
        else:
            ROI_gpd = gpd.read_file(region_of_interest)

        self.logger.info("Dissolving ROI")
        # Disolve to ensure there is only one row
        ROI_gpd = ROI_gpd.dissolve()
        self.logger.info("Setting CRS and buffering ROI")
        # Make sure we're using a projected CRS so a buffer can be applied
        ROI_gpd = ensure_projected_CRS(ROI_gpd)
        # Apply the buffer, plus the tolerance, to ensure we keep at least the requested region
        ROI_gpd["geometry"] = ROI_gpd.buffer(buffer_meters + simplify_tol_meters)
        # Simplify the geometry to reduce the computational load
        ROI_gpd.geometry = ROI_gpd.geometry.simplify(simplify_tol_meters)
        self.logger.info("Dissolving buffered ROI")
        # Disolve again in case
        ROI_gpd = ROI_gpd.dissolve()

        self.logger.info("Extracting verts for dataframe")
        # Get the vertices as a dataframe in the same CRS
        verts_df = self.get_verts_geodataframe(ROI_gpd.crs)
        self.logger.info("Checking intersection of verts with ROI")
        # Determine which vertices are within the ROI polygon
        verts_in_ROI = gpd.tools.overlay(verts_df, ROI_gpd, how="intersection")
        # Extract the IDs of the set within the polygon
        vert_inds = verts_in_ROI["vert_ID"].to_numpy()

        self.logger.info("Extracting points from pyvista mesh")
        # Extract a submesh using these IDs, which is returned as an UnstructuredGrid
        subset_unstructured_grid = self.pyvista_mesh.extract_points(vert_inds)
        self.logger.info("Extraction surface from subset mesh")
        # Convert the unstructured grid to a PolyData (mesh) again
        subset_mesh = subset_unstructured_grid.extract_surface()

        # If we need the indices into the original mesh, return those
        if return_original_IDs:
            try:
                point_IDs = subset_unstructured_grid["vtkOriginalPointIds"]
                face_IDs = subset_unstructured_grid["vtkOriginalCellIds"]
            except KeyError:
                point_IDs = np.array([])
                face_IDs = np.array([])

            return (
                subset_mesh,
                point_IDs,
                face_IDs,
            )
        # Else return just the mesh
        return subset_mesh

    def add_label(self, label_name, label_ID):
        if label_ID is not np.nan:
            self.IDs_to_labels[label_ID] = label_name

    def get_IDs_to_labels(self):
        # Convert to int type to avoid json serialization issues
        if self.IDs_to_labels is None:
            return None
        return {int(k): v for k, v in self.IDs_to_labels.items()}

    def get_label_names(self):
        self.logger.warning(
            "This method will be deprecated in favor of get_IDs_to_labels since it doesn't handle non-sequential indices"
        )
        if self.IDs_to_labels is None:
            return None
        return list(self.IDs_to_labels.values())

    # Vertex methods
    def get_vertices_in_CRS(
        self, output_CRS: pyproj.CRS, force_easting_northing: bool = True
    ):
        """Return the coordinates of the mesh vertices in a given CRS

        Args:
            output_CRS (pyproj.CRS): The coordinate reference system to transform to
            force_easting_northing (bool, optional): Ensure that the returned points are east first, then north

        Returns:
            np.ndarray: (n_points, 3)
        """
        # Reproject the mesh
        reprojected_mesh = self.reproject_CRS(output_CRS, inplace=False)
        verts_in_output_CRS = np.array(reprojected_mesh.points)

        # Pyproj respects the CRS axis ordering, which is northing/easting for most projected coordinate systems
        # This causes headaches because it's assumed by rasterio and geopandas to be easting/northing
        # https://rasterio.readthedocs.io/en/stable/api/rasterio.crs.html#rasterio.crs.epsg_treats_as_latlong
        if force_easting_northing and rio.crs.epsg_treats_as_latlong(output_CRS):
            # Swap first two columns
            verts_in_output_CRS = verts_in_output_CRS[:, [1, 0, 2]]

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

        df = pd.DataFrame(
            {
                "east": verts_in_geopolygon_crs[:, 0],
                "north": verts_in_geopolygon_crs[:, 1],
            }
        )
        # Create a column of Point objects to use as the geometry
        df["geometry"] = gpd.points_from_xy(df["east"], df["north"])
        points = gpd.GeoDataFrame(df, crs=crs)

        # Add an index column because the normal index will not be preserved in future operations
        points[VERT_ID] = df.index

        return points

    def get_faces_2d_gdf(
        self,
        crs: pyproj.CRS,
        include_3d_2d_ratio: bool = False,
        data_dict: dict = {},
        faces_mask: typing.Union[np.ndarray, None] = None,
        cache_data: bool = False,
    ) -> gpd.GeoDataFrame:
        """Get a geodataframe of triangles for the 2D projection of each face of the mesh

        Args:
            crs (pyproj.CRS):
                Coordinate reference system of the dataframe
            include_3d_2d_ratio (bool, optional):
                Compute the ratio of the 3D area of the face to the 2D area. This relates to the
                slope of the face relative to horizontal. The computed data will be stored in the
                column corresponding to the value of RATIO_3D_2D_KEY. Defaults to False.
            data_dict (dict, optional):
                Additional information to add to the dataframe. It must be a dict where the keys
                are the names of the columns and the data is a np.ndarray of n_faces elemenets.
                Defaults to {}.
            faces_mask (typing.Union[np.ndarray, None], optional):
                A binary mask corresponding to which faces to return. Used to improve runtime of
                creating the dataframe or downstream steps. Defaults to None.
            cache_data (bool):
                Whether to cache expensive results in memory as object attributes. Defaults to False.

        Returns:
            geopandas.GeoDataFrame: A dataframe for each triangular face
        """
        # Computing this data can be slow, and we might call it multiple times. This is especially
        # true for doing clustered polygon labeling
        if cache_data:
            mesh_hash = self.get_mesh_hash()
            faces_mask_hash = hash(
                faces_mask.tobytes() if faces_mask is not None else 0
            )
            # Create a key that uniquely identifies the relavant inputs
            cache_key = (mesh_hash, faces_mask_hash, crs)

            # See if the face polygons were in the cache. If not, None will be returned
            cached_values = self.face_polygons_cache.get(cache_key)
        else:
            cached_values = None

        if cached_values is not None:
            face_polygons, faces = cached_values
            logging.info("Using cached face polygons")
        else:
            self.logger.info("Computing faces in working CRS")
            # Get the mesh vertices in the desired export CRS
            verts_in_crs = self.get_vertices_in_CRS(crs)
            # Get a triangle in geospatial coords for each face
            # (n_faces, 3 points, xyz)
            faces = verts_in_crs[self.faces]

            # Select only the requested faces
            if faces_mask is not None:
                faces = faces[faces_mask]

            # Extract the first two columns and convert them to a list of tuples of tuples
            faces_2d_tuples = [tuple(map(tuple, a)) for a in faces[..., :2]]
            face_polygons = [
                Polygon(face_tuple)
                for face_tuple in tqdm(
                    faces_2d_tuples, desc=f"Converting faces to polygons"
                )
            ]
            self.logger.info("Creating dataframe of faces")

            if cache_data:
                # Save computed data to the cache for the future
                self.face_polygons_cache[cache_key] = (face_polygons, faces)

        # Remove data corresponding to masked faces
        if faces_mask is not None:
            data_dict = {k: v[faces_mask] for k, v in data_dict.items()}

        # Compute the ratio between the 3D area and the projected top-down 2D area
        if include_3d_2d_ratio:
            if cache_data:
                # Check if ratios are cached
                ratios = self.face_2d_3d_ratios_cache.get(cache_key)
            else:
                ratios = None

            # Ratios need to be computed
            if ratios is None:
                ratios = []
                for face in tqdm(faces, desc="Computing ratio of 3d to 2d area"):
                    area, area_2d = compute_3D_triangle_area(face)
                    ratios.append(area / area_2d)

                if cache_data:
                    self.face_2d_3d_ratios_cache[cache_key] = ratios

            # Add the ratios to the data dict
            data_dict[RATIO_3D_2D_KEY] = ratios

        # Create the dataframe
        faces_gdf = gpd.GeoDataFrame(
            data=data_dict,
            geometry=face_polygons,
            crs=crs,
        )

        return faces_gdf

    # Transform labels face<->vertex methods

    def face_to_vert_texture(self, face_IDs):
        """_summary_

        Args:
            face_IDs (np.array): (n_faces,) The integer IDs of the faces
        """
        raise NotImplementedError()
        # TODO figure how to have a NaN class that
        for i in tqdm(range(self.pyvista_mesh.points.shape[0])):
            # Find which faces are using this vertex
            matching = np.sum(self.faces == i, axis=1)
            # matching_inds = np.where(matching)[0]
            # matching_IDs = face_IDs[matching_inds]
            # most_common_ind = Counter(matching_IDs).most_common(1)

    def vert_to_face_texture(self, vert_IDs, discrete=True):
        if vert_IDs is None:
            raise ValueError("None")

        vert_IDs = np.squeeze(vert_IDs)
        if vert_IDs.ndim != 1 and discrete:
            raise ValueError(
                f"Can only perform discrete conversion with one dimensional array but instead had {vert_IDs.ndim}"
            )

        # Each row contains the IDs of each vertex
        values_per_face = vert_IDs[self.faces]
        if discrete:
            # Now we need to "vote" for the best one
            max_ID = np.nanmax(vert_IDs)
            # This means that all textures are nans
            if not np.isfinite(max_ID):
                self.logger.warn(
                    "In vertex to face texture conversion, all nans encountered"
                )
                # Return all nans
                return np.full(values_per_face.shape[0], fill_value=np.nan)

            # Compute the most common class per face, doing so by batch to avoid OOM. This is because
            # this implementation creates a (n_faces, n_classes) array which can grow quite large
            chunk_size = 100000
            most_common_class_per_face = np.concatenate(
                [
                    fair_mode_non_nan(values_per_face[i : i + chunk_size])
                    for i in tqdm(
                        range(0, values_per_face.shape[0], chunk_size),
                        desc="Computing most common class per face by batch",
                    )
                ],
                axis=0,
            )

            return most_common_class_per_face
        else:
            average_value_per_face = np.mean(values_per_face, axis=1)
            return average_value_per_face

    # Operations on vector data
    def get_values_for_verts_from_vector(
        self,
        vector_source: typing.Union[gpd.GeoDataFrame, PATH_TYPE],
        column_names: typing.Union[str, typing.List[str]],
    ) -> np.ndarray:
        """Get the value from a dataframe for each vertex

        Args:
            vector_source (typing.Union[gpd.GeoDataFrame, PATH_TYPE]): geo data frame or path to data that can be loaded by geopandas
            column_names (typing.Union[str, typing.List[str]]): Which columns to obtain data from

        Returns:
            np.ndarray | dict[str, np.ndarray]:
                An array or dict of string->array mappings, with one element per vector file polygon
            np.ndarray | dict[str, np.ndarray]:
                An array or dict of string->array mappings, with one element per mesh vertex
        """
        # Lead the vector data if not already provided in memory
        if isinstance(vector_source, gpd.GeoDataFrame):
            gdf = vector_source
        else:
            # This will error if not readable
            gdf = gpd.read_file(vector_source)

        # Infer or standardize the column names
        if column_names is None:
            # Check if there is only one real column
            if len(gdf.columns) == 2:
                column_names = list(filter(lambda x: x != "geometry", gdf.columns))
            else:
                # Log as well since this may be caught by an exception handler,
                # and it's a user error that can be corrected
                self.logger.error(
                    "No column name provided and ambigious which column to use"
                )
                raise ValueError(
                    "No column name provided and ambigious which column to use"
                )
        # If only one column is provided, make it a one-length list
        elif isinstance(column_names, str):
            column_names = [column_names]

        # Get a dataframe of vertices
        verts_df = self.get_verts_geodataframe(gdf.crs)

        # See which vertices are in the geopolygons
        points_in_polygons_gdf = gpd.tools.overlay(verts_df, gdf, how="intersection")
        # Get the index array
        index_array = points_in_polygons_gdf[VERT_ID].to_numpy()

        # This is one entry per vertex
        labeled_verts_dict = {}
        all_values_dict = {}
        # Extract the data from each
        for column_name in column_names:
            # Create an array corresponding to all the points and initialize to NaN
            column_values = points_in_polygons_gdf[column_name]
            # TODO clean this up
            if column_values.dtype == str or column_values.dtype == np.dtype("O"):
                # TODO be set to the default value for the type of the column
                null_value = "null"
            elif column_values.dtype == int:
                null_value = 255
            else:
                null_value = np.nan
            # Create an array, one per vertex, with the null value
            values = np.full(
                shape=verts_df.shape[0],
                dtype=column_values.dtype,
                fill_value=null_value,
            )
            # Assign the labeled values
            values[index_array] = column_values

            # Record the results
            labeled_verts_dict[column_name] = values
            all_values_dict[column_name] = gdf[column_name]

        # If only one name was requested, just return that
        if len(column_names) == 1:
            labeled_verts = np.array(list(labeled_verts_dict.values())[0])
            all_values = np.array(list(all_values_dict.values())[0])

            return labeled_verts, all_values
        # Else return a dict of all requested values
        return labeled_verts_dict, all_values_dict

    def save_IDs_to_labels(self, savepath: PATH_TYPE):
        """saves the contents of the IDs_to_labels to the file savepath provided

        Args:
            savepath (PATH_TYPE): path to the file where the data must be saved
        """

        # Save the classes filename
        ensure_containing_folder(savepath)
        if self.is_discrete_texture():
            self.logger.info("discrete texture, saving classes")
            self.logger.info(f"Saving IDs_to_labels to {str(savepath)}")
            with open(savepath, "w") as outfile_h:
                json.dump(
                    self.get_IDs_to_labels(), outfile_h, ensure_ascii=False, indent=4
                )
        else:
            self.logger.warn("non-discrete texture, not saving classes")

    def save_mesh(self, savepath: PATH_TYPE, save_vert_texture: bool = True):
        # TODO consider moving most of this functionality to a utils file
        if save_vert_texture:
            vert_texture = self.get_texture(request_vertex_texture=True)
            n_channels = vert_texture.shape[1]

            if n_channels == 1:
                vert_texture = np.nan_to_num(vert_texture, nan=NULL_TEXTURE_INT_VALUE)
                vert_texture = np.tile(vert_texture, reps=(1, 3))
            if n_channels > 3:
                self.logger.warning(
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

        # Create folder if it doesn't exist
        ensure_containing_folder(savepath)
        # Actually save the mesh
        self.pyvista_mesh.save(savepath, texture=vert_texture)
        self.save_IDs_to_labels(Path(savepath).stem + "_IDs_to_labels.json")

    def label_polygons(
        self,
        face_labels: np.ndarray,
        polygons: typing.Union[PATH_TYPE, gpd.GeoDataFrame],
        face_weighting: typing.Union[None, np.ndarray] = None,
        sjoin_overlay: bool = True,
        return_class_labels: bool = True,
        unknown_class_label: str = "unknown",
        buffer_dist_meters: float = 2.0,
    ):
        """Assign a class label to polygons using labels per face

        Args:
            face_labels (np.ndarray): (n_faces,) array of integer labels
            polygons (typing.Union[PATH_TYPE, gpd.GeoDataFrame]): Geospatial polygons to be labeled
            face_weighting (typing.Union[None, np.ndarray], optional):
                (n_faces,) array of scalar weights for each face, to be multiplied with the
                contribution of this face. Defaults to None.
            sjoin_overlay (bool, optional):
                Whether to use `gpd.sjoin` or `gpd.overlay` to compute the overlay. Sjoin is
                substaintially faster, but only uses mesh faces that are entirely within the bounds
                of the polygon, rather than computing the intersecting region for
                partially-overlapping faces. Defaults to True.
            return_class_labels: (bool, optional):
                Return string representation of class labels rather than float. Defaults to True.
            unknown_class_label (str, optional):
                Label for predicted class for polygons with no overlapping faces. Defaults to "unknown".
            buffer_dist_meters: (Union[float, None], optional)
                Only applicable if sjoin_overlay=False. In that case, include faces entirely within
                the region that is this distance in meters from the polygons. Defaults to 2.0.

        Raises:
            ValueError: if faces_labels or face_weighting is not 1D

        Returns:
            list(typing.Union[str, int]):
                (n_polygons,) list of labels. Either float values, represnting integer IDs or nan,
                or string values representing the class label
        """
        # Premptive error checking before expensive operations
        face_labels = np.squeeze(face_labels)
        if face_labels.ndim != 1:
            raise ValueError(
                f"Faces labels must be one-dimensional, but is {face_labels.ndim}"
            )
        if face_weighting is not None:
            face_weighting = np.squeeze(face_weighting)
            if face_weighting.ndim != 1:
                raise ValueError(
                    f"Faces labels must be one-dimensional, but is {face_weighting.ndim}"
                )

        # Ensure that the input is a geopandas dataframe
        polygons_gdf = ensure_projected_CRS(coerce_to_geoframe(polygons))
        # Extract just the geometry
        polygons_gdf = polygons_gdf[["geometry"]]

        # Only get faces for which there is a non-nan label. Otherwise it is just additional compute
        faces_mask = np.isfinite(face_labels)

        # Get the faces of the mesh as a geopandas dataframe
        # Include the predicted face labels as a column in the dataframe
        faces_2d_gdf = self.get_faces_2d_gdf(
            polygons_gdf.crs,
            include_3d_2d_ratio=True,
            data_dict={CLASS_ID_KEY: face_labels},
            faces_mask=faces_mask,
            cache_data=True,
        )

        # If a per-face weighting is provided, multiply that with the 3d to 2d ratio
        if face_weighting is not None:
            face_weighting = face_weighting[faces_mask]
            faces_2d_gdf["face_weighting"] = (
                faces_2d_gdf[RATIO_3D_2D_KEY] * face_weighting
            )
        # If not, just use the ratio
        else:
            faces_2d_gdf["face_weighting"] = faces_2d_gdf[RATIO_3D_2D_KEY]

        # Set the precision to avoid approximate coliniearity errors
        faces_2d_gdf.geometry = shapely.set_precision(
            faces_2d_gdf.geometry.values, 1e-6
        )
        polygons_gdf.geometry = shapely.set_precision(
            polygons_gdf.geometry.values, 1e-6
        )

        # Set the ID field so it's available after the overlay operation
        # Note that polygons_gdf.index is a bad choice, because this df could be a subset of another
        # one and the index would not start from 0
        polygons_gdf["polygon_ID"] = np.arange(len(polygons_gdf))

        # Since overlay is expensive, we first discard faces that are not near the polygons

        # Dissolve the polygons to form one ROI
        merged_polygons = polygons_gdf.dissolve()
        # Try to decrease the number of elements in the polygon by expanding
        # and then simplifying the number of elements in the polygon
        merged_polygons.geometry = merged_polygons.buffer(buffer_dist_meters)
        merged_polygons.geometry = merged_polygons.simplify(buffer_dist_meters)

        # Determine which face IDs intersect the ROI. This is slow
        start = time()
        self.logger.info("Starting to subset to ROI")

        # Check which faces are fully within the buffered regions around the query polygons
        # Note that using sjoin has been faster than any other approach I've tried, despite seeming
        # to compute more information than something like gpd.within
        contained_faces = gpd.sjoin(
            faces_2d_gdf, merged_polygons, how="left", predicate="within"
        )["index_right"].notna()
        faces_2d_gdf = faces_2d_gdf.loc[contained_faces]
        self.logger.info(f"Subset to ROI in {time() - start} seconds")

        start = time()
        self.logger.info("Starting `overlay`")
        if sjoin_overlay:
            overlay = gpd.sjoin(
                faces_2d_gdf, polygons_gdf, how="left", predicate="within"
            )
            self.logger.info(f"Overlay time with gpd.sjoin: {time() - start}")
        else:
            # Drop faces not included
            overlay = polygons_gdf.overlay(
                faces_2d_gdf, how="identity", keep_geom_type=False
            )
            self.logger.info(f"Overlay time with gpd.overlay: {time() - start}")

        # Drop nan, for geometries that don't intersect the polygons
        overlay.dropna(inplace=True)
        # Compute the weighted area for each face, which may have been broken up by the overlay
        overlay["weighted_area"] = overlay.area * overlay["face_weighting"]

        # Extract only the neccessary columns
        overlay = overlay.loc[:, ["polygon_ID", CLASS_ID_KEY, "weighted_area"]]
        aggregated_data = overlay.groupby(["polygon_ID", CLASS_ID_KEY]).agg(np.sum)
        # Compute the highest weighted class prediction
        # Modified from https://stackoverflow.com/questions/27914360/python-pandas-idxmax-for-multiple-indexes-in-a-dataframe
        max_rows = aggregated_data.loc[
            aggregated_data.groupby(["polygon_ID"], sort=False)[
                "weighted_area"
            ].idxmax()
        ].reset_index()

        # Make the class predictions a list of IDs with nans where no information is available
        pred_subset_IDs = max_rows[CLASS_ID_KEY].to_numpy(dtype=float)
        pred_subset_IDs[max_rows["weighted_area"].to_numpy() == 0] = np.nan

        predicted_class_IDs = np.full(len(polygons_gdf), np.nan)
        predicted_class_IDs[max_rows["polygon_ID"].to_numpy(dtype=int)] = (
            pred_subset_IDs
        )
        predicted_class_IDs = predicted_class_IDs.tolist()

        # Post-process to string label names if requested and IDs_to_labels exists
        if return_class_labels and (
            (IDs_to_labels := self.get_IDs_to_labels()) is not None
        ):
            # convert the IDs into labels
            # Any label marked as nan is set to the unknown class label, since we had no predictions for it
            predicted_class_IDs = [
                (IDs_to_labels[int(pi)] if np.isfinite(pi) else unknown_class_label)
                for pi in predicted_class_IDs
            ]
        return predicted_class_IDs

    def export_face_labels_vector(
        self,
        face_labels: typing.Union[np.ndarray, None] = None,
        export_file: PATH_TYPE = None,
        export_crs: pyproj.CRS = LAT_LON_CRS,
        label_names: typing.Tuple = None,
        ensure_non_overlapping: bool = False,
        simplify_tol: float = 0.0,
        drop_nan: bool = True,
        vis: bool = True,
        batched_unary_union_kwargs: typing.Dict = {
            "batch_size": 500000,
            "sort_by_loc": True,
            "grid_size": 0.05,
            "simplify_tol": 0.05,
        },
        vis_kwargs: typing.Dict = {},
    ) -> gpd.GeoDataFrame:
        """Export the labels for each face as a on-per-class multipolygon

        Args:
            face_labels (np.ndarray):
                This can either be a 1- or 2-D array. If 1-D, it is (n_faces,) where each element
                is an integer class label for that face. If 2-D, it's (n_faces, n_classes) and a
                nonzero element at (i, j) represents a class prediction for the ith faces and jth
                class
            export_file (PATH_TYPE, optional):
                Where to export. The extension must be a filetype that geopandas can write.
                Defaults to None, if unset, nothing will be written.
            export_crs (pyproj.CRS, optional): What CRS to export in.. Defaults to pyproj.CRS.from_epsg(4326), lat lon.
            label_names (typing.Tuple, optional): Optional names, that are indexed by the labels. Defaults to None.
            ensure_non_overlapping (bool, optional): Should regions where two classes are predicted at different z heights be assigned to one class
            simplify_tol: (float, optional): Tolerence in meters to use to simplify geometry
            drop_nan (bool, optional): Don't export the nan class, often used for background
            vis: should the result be visualzed
            batched_unary_union_kwargs (dict, optional): Keyword arguments for batched_unary_union_call
            vis_kwargs: keyword argmument dict for visualization

        Raises:
            ValueError: If the wrong number of faces labels are provided

        Returns:
            gpd.GeoDataFrame: Merged data
        """
        # Compute the working projected CRS
        # This is important because having things in meters makes things easier
        self.logger.info("Computing working CRS")
        lon, lat, _ = self.get_vertices_in_CRS(output_CRS=LAT_LON_CRS)[0]
        working_CRS = get_projected_CRS(lon=lon, lat=lat)

        # Try to extract face labels if not set
        if face_labels is None:
            face_labels = self.get_texture(request_vertex_texture=False)

        # Check that the correct number of labels are provided
        if face_labels.shape[0] != self.faces.shape[0]:
            raise ValueError()

        # Get the geospatial faces dataframe
        faces_gdf = self.get_faces_2d_gdf(crs=working_CRS)

        self.logger.info("Creating dataframe of multipolygons")

        # Check how the data is represented, as a 1-D list of integers or one/many-hot encoding
        face_labels_is_2d = face_labels.ndim == 2 and face_labels.shape[1] != 1
        if face_labels_is_2d:
            # Non-null columns
            unique_IDs = np.nonzero(np.sum(face_labels, axis=0))[1]
        else:
            face_labels = np.squeeze(face_labels)
            unique_IDs = np.unique(face_labels)

        if drop_nan:
            # Drop nan from the list of IDs
            unique_IDs = unique_IDs[np.isfinite(unique_IDs)]
        multipolygon_list = []
        # For each unique ID, aggregate all the faces together
        # This is the same as geopandas.groupby, but that is slow and can out of memory easily
        # due to the large number of polygons
        # Instead, we replace the default shapely.unary_union with our batched implementation
        for unique_ID in tqdm(unique_IDs, desc="Merging faces for each class"):
            if face_labels_is_2d:
                # Nonzero elements of the column
                matching_face_mask = face_labels[:, unique_ID] > 0
            else:
                # Elements that match the ID in question
                matching_face_mask = face_labels == unique_ID
            matching_face_inds = np.nonzero(matching_face_mask)[0]
            matching_face_polygons = faces_gdf.iloc[matching_face_inds]
            list_of_polygons = matching_face_polygons.geometry.values
            multipolygon = batched_unary_union(
                list_of_polygons, **batched_unary_union_kwargs
            )
            multipolygon_list.append(multipolygon)

        working_gdf = gpd.GeoDataFrame(
            {CLASS_ID_KEY: unique_IDs}, geometry=multipolygon_list, crs=working_CRS
        )

        if label_names is not None:
            names = [
                (label_names[int(ID)] if np.isfinite(ID) else "nan")
                for ID in working_gdf[CLASS_ID_KEY]
            ]
            working_gdf[CLASS_NAMES_KEY] = names

        # Simplify the output geometry
        if simplify_tol > 0.0:
            self.logger.info("Running simplification")
            working_gdf.geometry = working_gdf.geometry.simplify(simplify_tol)

        # Make sure that the polygons are non-overlapping
        if ensure_non_overlapping:
            # TODO create a version that tie-breaks based on the number of predicted faces for each
            # class and optionally the ratios of 3D to top-down areas for the input triangles.
            self.logger.info("Ensuring non-overlapping polygons")
            working_gdf = ensure_non_overlapping_polygons(working_gdf)

        # Transform from the working crs to export crs
        export_gdf = working_gdf.to_crs(export_crs)

        # Export if a file is provided
        if export_file is not None:
            ensure_containing_folder(export_file)
            export_gdf.to_file(export_file)

        # Vis if requested
        if vis:
            self.logger.info("Plotting")
            export_gdf.plot(
                column=CLASS_NAMES_KEY if label_names is not None else CLASS_ID_KEY,
                aspect=1,
                legend=True,
                **vis_kwargs,
            )
            plt.show()

        return export_gdf

    # Operations on raster files

    def get_vert_values_from_raster_file(
        self,
        raster_file: PATH_TYPE,
        return_verts_in_CRS: bool = False,
        nodata_fill_value: float = np.nan,
    ):
        """Compute the height above groun for each point on the mesh

        Args:
            raster_file (PATH_TYPE, optional): The path to the geospatial raster file.
            return_verts_in_CRS (bool, optional): Return the vertices transformed into the raster CRS
            nodata_fill_value (float, optional): Set data defined by the opened file as NODATAVAL to this value

        Returns:
            np.ndarray: samples from raster. Either (n_verts,) or (n_verts, n_raster_channels)
            np.ndarray (optional): (n_verts, 3) the vertices in the raster CRS
        """
        # Open the DTM file
        raster = rio.open(raster_file)
        # Get the mesh points in the coordinate reference system of the DTM
        verts_in_raster_CRS = self.get_vertices_in_CRS(
            raster.crs, force_easting_northing=True
        )

        # Get the points as a list
        easting_points = verts_in_raster_CRS[:, 0].tolist()
        northing_points = verts_in_raster_CRS[:, 1].tolist()

        # Zip them together
        zipped_locations = zip(easting_points, northing_points)
        sampling_iter = tqdm(
            zipped_locations,
            desc=f"Sampling values from raster {raster_file}",
            total=verts_in_raster_CRS.shape[0],
        )
        # Sample the raster file and squeeze if single channel
        sampled_raster_values = np.squeeze(np.array(list(raster.sample(sampling_iter))))

        # Set nodata locations to nan
        # TODO figure out if it will ever be a problem to take the first value
        sampled_raster_values[sampled_raster_values == raster.nodatavals[0]] = (
            nodata_fill_value
        )

        if return_verts_in_CRS:
            return sampled_raster_values, verts_in_raster_CRS

        return sampled_raster_values

    def get_height_above_ground(
        self, DTM_file: PATH_TYPE, threshold: float = None
    ) -> np.ndarray:
        """Return height above ground for a points in the mesh and a given DTM

        Args:
            DTM_file (PATH_TYPE): Path to the digital terrain model raster
            threshold (float, optional):
                If not None, return a boolean mask for points under this height. Defaults to None.

        Returns:
            np.ndarray: Either the height above ground or a boolean mask for ground points
        """
        # Get the height from the DTM and the points in the same CRS
        DTM_heights, verts_in_raster_CRS = self.get_vert_values_from_raster_file(
            DTM_file, return_verts_in_CRS=True
        )
        # Extract the vertex height as the third channel
        verts_height = verts_in_raster_CRS[:, 2]
        # Subtract the two to get the height above ground
        height_above_ground = verts_height - DTM_heights

        # If the threshold is not None, return a boolean mask that is true for ground points
        if threshold is not None:
            # Return boolean mask
            # TODO see if this will break for nan values
            return height_above_ground < threshold
        # Return height above ground
        return height_above_ground

    def label_ground_class(
        self,
        DTM_file: PATH_TYPE,
        height_above_ground_threshold: float,
        labels: typing.Union[None, np.ndarray] = None,
        only_label_existing_labels: bool = True,
        ground_class_name: str = "ground",
        ground_ID: typing.Union[None, int] = None,
        set_mesh_texture: bool = False,
    ) -> np.ndarray:
        """
        Set vertices to a potentially-new class with a thresholded height above the DTM.
        TODO, consider handling face textures as well

        Args:
            DTM_file (PATH_TYPE): Path to the DTM file
            height_above_ground_threshold (float): Height (meters) above that DTM that points below are considered ground
            labels (typing.Union[None, np.ndarray], optional): Vertex texture, otherwise will be queried from mesh. Defaults to None.
            only_label_existing_labels (bool, optional): Only label points that already have non-null labels. Defaults to True.
            ground_class_name (str, optional): The potentially-new ground class name. Defaults to "ground".
            ground_ID (typing.Union[None, int], optional): What value to use for the ground class. Will be set inteligently if not provided. Defaults to None.

        Returns:
            np.ndarray: The updated labels
        """

        if labels is None:
            # Default to using vertex labels since it's the native way to check height above the DTM
            use_vertex_labels = True
        elif labels is not None:
            # Check the size of the input labels and set what type they are. Note this could override existing value
            if labels.shape[0] == self.pyvista_mesh.points.shape[0]:
                use_vertex_labels = True
            elif labels.shape[0] == self.faces.shape[0]:
                use_vertex_labels = False
            else:
                raise ValueError(
                    "Labels were provided but didn't match the shape of vertices or faces"
                )

        # if a labels are not provided, get it from the mesh
        if labels is None:
            # Get the vertex textures from the mesh
            labels = self.get_texture(
                request_vertex_texture=use_vertex_labels,
            )

        # Compute which vertices are part of the ground by thresholding the height above the DTM
        ground_mask = self.get_height_above_ground(
            DTM_file=DTM_file, threshold=height_above_ground_threshold
        )
        # If we needed a mask for the faces, compute that instead
        if not use_vertex_labels:
            ground_mask = self.vert_to_face_texture(ground_mask.astype(int)).astype(
                bool
            )

        # Replace only vertices that were previously labeled as something else, to avoid class imbalance
        if only_label_existing_labels:
            # Find which vertices are labeled
            is_labeled = np.isfinite(labels[:, 0])
            # Find which points are ground that were previously labeled as something else
            ground_mask = np.logical_and(is_labeled, ground_mask)

        # Get the existing label names
        IDs_to_labels = self.get_IDs_to_labels()

        if IDs_to_labels is None and ground_ID is None:
            # This means that the label is continous, so the concept of ID is meaningless
            ground_ID = np.nan
        elif IDs_to_labels is not None and ground_class_name in IDs_to_labels.values():
            # If the ground class name is already in the list, set newly-predicted vertices to that class
            # Get the dictionary mapping in the reverse direction
            labels_to_IDs = {v: k for k, v in IDs_to_labels.items()}
            # Determine the ID corresponding to the ground class name
            ground_ID = labels_to_IDs.get(ground_class_name)
        elif IDs_to_labels is not None:
            # If the label names are present, and the class is not already included, add it as the last element
            if ground_ID is None:
                # Set it to the first unused ID
                # TODO improve this since it should be the max plus one
                ground_ID = len(IDs_to_labels)

        self.add_label(label_name=ground_class_name, label_ID=ground_ID)

        # Replace mask for ground_vertices
        labels[ground_mask, 0] = ground_ID

        # Optionally apply the texture to the mesh
        if set_mesh_texture:
            # TODO look into why this shouldn't update the IDs to labels.
            # I guess because it may have been initially user-provided.
            self.set_texture(labels, update_IDs_to_labels=False)

        return labels

    def get_mesh_hash(self):
        """Generates a hash value for the mesh based on its points and faces
        Returns:
            int: A hash value representing the current mesh.
        """
        hasher = hashlib.sha256()
        hasher.update(self.pyvista_mesh.points.tobytes())
        hasher.update(self.pyvista_mesh.faces.tobytes())
        return hasher.hexdigest()

    def get_mesh_in_cameras_coords(
        self,
        cameras: typing.Union[PhotogrammetryCamera, PhotogrammetryCameraSet],
        inplace: bool = False,
    ) -> typing.Optional[pv.PolyData]:
        """Obtain a mesh in the same local coordinate frame convention as the camera set.

        Args:
            cameras (typing.Union[PhotogrammetryCamera, PhotogrammetryCameraSet]):
                Camera or camera set to match the convention of.
            inplace (bool, optional):
                Should the object be update to reflect this convetion. Otherwise, a pyvista mesh is
                returned and the self object remains unchanged. Defaults to False.

        Returns:
            typing.Optional[pv.PolyData]: Mesh in the camera's coordinate system if inplace=False
        """
        # Reproject the mesh into ECEF
        mesh = self.reproject_CRS(EARTH_CENTERED_EARTH_FIXED_CRS, inplace=False)

        # Get the inverse 4x4 transform, which maps from Earth Centered, Earth Fixed (EPSG:4978)
        # to the coordinates that the cameras are in
        local_to_epsg_4978_transform = cameras.get_local_to_epsg_4978_transform()
        epsg_4978_to_camera = np.linalg.inv(local_to_epsg_4978_transform)
        # Transform the mesh using this transform
        mesh = mesh.transform(epsg_4978_to_camera, inplace=False)

        if inplace:
            # Overwrite the mesh with the updated version
            self.pyvista_mesh = mesh
            # Indicate that there is no longer a valid CRS
            self.CRS = None
            # Return None to match common convention for inplace methods
            return None

        return mesh

    def pix2face(
        self,
        cameras: typing.Union[PhotogrammetryCamera, PhotogrammetryCameraSet],
        mesh: typing.Optional[pv.PolyData] = None,
        render_img_scale: float = 1,
        save_to_cache: bool = False,
        cache_folder: typing.Union[None, PATH_TYPE] = CACHE_FOLDER,
        distortion_set: typing.Optional[PhotogrammetryCameraSet] = None,
        apply_distortion: bool = True,
    ) -> np.ndarray:
        """Compute the face that a ray from each pixel would intersect for each camera

        Args:
            cameras (typing.Union[PhotogrammetryCamera, PhotogrammetryCameraSet]):
                A single camera or set of cameras. For each camera, the correspondences between
                pixels and the face IDs of the mesh will be computed. The images of all cameras
                are assumed to be the same size.
            render_img_scale (float, optional):
                Create a pix2face map that is this fraction of the original image scale. Defaults
                to 1.
            save_to_cache (bool, optional):
                Should newly-computed values be saved to the cache. This may speed up future operations
                but can take up 100s of GBs of space. Defaults to False.
            cache_folder ((PATH_TYPE, None), optional):
                Where to check for and save to cached data. Only applicable if use_cache=True.
                Defaults to CACHE_FOLDER
            distortion_set (PhotogrammetryCameraSet, optional): camera set used for calculating
                and caching the distortion/undistortion maps. This is only required if apply_distortion
                is True. Note that if you are calling pix2face on batches, you should pass the
                full camera set in as the distortion_set so that the maps are cached once.
            apply_distortion (bool, optional):
                Should distortion correction be applied. This is expensive but can be neccesary for
                some camera models which significantly differ from a pinhole model.

        Returns:
            np.ndarray: For each camera, there is an array that is the shape of an image and
            contains the integer face index for the ray originating at that pixel. Any pixel for
            which the given ray does not intersect a face is given a value of -1. If the input is
            a single PhotogrammetryCamera, the shape is (h, w). If it's a camera set, then it is
            (n_cameras, h, w). Note that a one-length camera set will have a leading singleton dim.
        """

        # Create a local mesh if it hasn't been created yet
        if mesh is None:
            mesh = self.get_mesh_in_cameras_coords(cameras)

        # If a set of cameras is passed in, call this method on each camera and concatenate
        # Other derived methods might be able to compute a batch of renders at once, but pyvista
        # cannot as far as I can tell.
        # Note that all inputs to pix2face need to be replicated here or those features won't be
        # passed on.
        if isinstance(cameras, PhotogrammetryCameraSet):
            pix2face_list = [
                self.pix2face(
                    cameras=camera,
                    mesh=mesh,
                    render_img_scale=render_img_scale,
                    save_to_cache=save_to_cache,
                    cache_folder=cache_folder,
                    distortion_set=distortion_set,
                    apply_distortion=apply_distortion,
                )
                for camera in cameras
            ]
            pix2face = np.stack(pix2face_list, axis=0)
            return pix2face

        ## Single camera case

        # Check if the cache contains a valid pix2face for the camera based on the dependencies
        # Compute hashes for the mesh and camera to unique identify mesh+camera pair
        # The cache will generate a unique key for each combination of the dependencies
        # If the cache generated key matches a cache file on disk, pix2face will be filled with the correct correspondance
        # If no match is found, recompute pix2face
        # If there’s an error loading the cached data, then clear the cache's contents, signified by on_error='clear'
        mesh_hash = self.get_mesh_hash()
        camera_hash = cameras.get_camera_hash()
        cacher = ub.Cacher(
            "pix2face",
            depends=[mesh_hash, camera_hash, render_img_scale],
            dpath=cache_folder,
            verbose=0,
        )
        pix2face = cacher.tryload(on_error="clear")
        ## Cache is valid
        if pix2face is not None:
            return pix2face

        # This needs to be an attribute of the class because creating a large number of plotters
        # results in an un-fixable memory leak.
        # See https://github.com/pyvista/pyvista/issues/2252
        # The first step is to clear it
        self.pix2face_plotter.clear()
        # This is important so there aren't intermediate values
        self.pix2face_plotter.disable_anti_aliasing()
        # Set the camera to the corresponding viewpoint
        self.pix2face_plotter.camera = cameras.get_pyvista_camera()

        ## Compute the base 256 encoding of the face ID
        n_faces = self.faces.shape[0]
        ID_values = np.arange(n_faces)

        # determine how many channels will be required to represent the number of faces
        n_channels = int(np.ceil(np.emath.logn(256, n_faces))) if n_faces != 0 else 0
        channel_multipliers = [256**i for i in range(n_channels)]

        # Compute the encoding of each value, least significant value first
        base_256_encoding = [
            np.mod(np.floor(ID_values / m).astype(int), 256)
            for m in channel_multipliers
        ]

        # ensure that there's a multiple of three channels
        n_padding = int(np.ceil(n_channels / 3.0) * 3 - n_channels)
        base_256_encoding.extend([np.zeros(n_faces)] * n_padding)

        # Assume that all images are the same size
        image_size = cameras.get_image_size(image_scale=render_img_scale)

        # Initialize pix2face
        pix2face = np.zeros(image_size, dtype=int)
        # Iterate over three-channel chunks. Each will be encoded as RGB and rendered
        for chunk_ind in range(int(len(base_256_encoding) / 3)):
            chunk_scalars = np.stack(
                base_256_encoding[3 * chunk_ind : 3 * (chunk_ind + 1)], axis=1
            ).astype(np.uint8)
            # Add the mesh with the associated scalars
            self.pix2face_plotter.add_mesh(
                mesh,
                scalars=chunk_scalars.copy(),
                rgb=True,
                diffuse=0.0,
                ambient=1.0,
            )

            # Perform rendering, this is the slow step
            rendered_img = self.pix2face_plotter.screenshot(
                window_size=(image_size[1], image_size[0]),
            )
            # Take the rendered values and interpret them as the encoded value
            # Make sure to not try to interpret channels that are not used in the encoding
            channels_to_decode = min(3, len(channel_multipliers) - 3 * chunk_ind)
            for i in range(channels_to_decode):
                channel_multiplier = channel_multipliers[chunk_ind * 3 + i]
                channel_value = (rendered_img[..., i] * channel_multiplier).astype(int)
                pix2face += channel_value

        # Mask out pixels for which the mesh was not visible
        # This is because the background will render as white
        # If there happen to be an exact power of (256^3) number of faces, the last one may get
        # erronously masked. This seems like a minimal concern but it could be addressed by adding
        # another channel or something like that
        pix2face[pix2face > n_faces] = -1

        if save_to_cache:
            # Save the most recently computed pix2face correspondance in the cache
            cacher.save(pix2face)

        if apply_distortion:
            # Warp the pix2face mask so it matches the warping of the real image which does not
            # conform to the pinhole model.
            # Note this step is slow, and especially on the first iteration which may take multiple
            # minutes.
            pix2face = distortion_set.warp_dewarp_image(
                camera=cameras,
                input_image=pix2face,
                warped_to_ideal=False,
                fill_value=-1,
                interpolation_order=0,  # nearest neighbor interpolation
                image_scale=render_img_scale,
            )

        return pix2face

    def render_flat(
        self,
        cameras: typing.Union[PhotogrammetryCamera, PhotogrammetryCameraSet],
        batch_size: int = 1,
        render_img_scale: float = 1,
        return_camera: bool = False,
        **pix2face_kwargs,
    ):
        """
        Render the texture from the viewpoint of each camera in cameras. Note that this is a
        generator so if you want to actually execute the computation, call list(*) on the output

        Args:
            cameras (typing.Union[PhotogrammetryCamera, PhotogrammetryCameraSet]):
                Either a single camera or a camera set. The texture will be rendered from the
                perspective of each one
            batch_size (int, optional):
                The batch size for pix2face. Defaults to 1.
            render_img_scale (float, optional):
                The rendered image will be this fraction of the original image corresponding to the
                virtual camera. Defaults to 1.
            return_camera (bool, optional):
                Should the camera be yielded as the second value

        Raises:
            TypeError: If cameras is not the correct type

        Yields:
            np.ndarray:
               The pix2face array for the next camera. The shape is
               (int(img_h*render_img_scale), int(img_w*render_img_scale)).
        """
        # Create a local mesh
        mesh = self.get_mesh_in_cameras_coords(cameras)

        if isinstance(cameras, PhotogrammetryCamera):
            # Construct a camera set of length one
            cameras = PhotogrammetryCameraSet([cameras])
        elif not isinstance(cameras, PhotogrammetryCameraSet):
            raise TypeError()

        # Get the face texture from the mesh
        # TODO consider whether the user should be able to pass a texture to this method. It could
        # make the user's life easier but makes this method more complex
        face_texture = self.get_texture(
            request_vertex_texture=False, try_verts_faces_conversion=True
        )
        texture_dim = face_texture.shape[1]

        # Iterate over batch of the cameras
        batch_stop = max(len(cameras) - batch_size + 1, 1)
        for batch_start in range(0, batch_stop, batch_size):
            batch_end = batch_start + batch_size
            batch_cameras = cameras[batch_start:batch_end]
            # Compute a batch of pix2face correspondences. This is likely the slowest step
            batch_pix2face = self.pix2face(
                cameras=batch_cameras,
                mesh=mesh,
                render_img_scale=render_img_scale,
                **pix2face_kwargs,
            )

            # Iterate over the batch dimension
            for i, pix2face in enumerate(batch_pix2face):
                # Record the original shape of the image
                img_shape = pix2face.shape[:2]
                # Flatten for indexing
                pix2face = pix2face.flatten()
                # Compute which pixels intersected the mesh
                mesh_pixel_inds = np.where(pix2face != -1)[0]
                # Initialize and all-nan array
                rendered_flattened = np.full(
                    (pix2face.shape[0], texture_dim), fill_value=np.nan
                )
                # Fill the values for which correspondences exist
                rendered_flattened[mesh_pixel_inds] = face_texture[
                    pix2face[mesh_pixel_inds]
                ]
                # reshape to an image, where the last dimension is the texture dimension
                rendered_img = rendered_flattened.reshape(img_shape + (texture_dim,))

                if return_camera:
                    yield (rendered_img, batch_cameras[i])
                else:
                    yield rendered_img

    def project_images(
        self,
        cameras: typing.Union[PhotogrammetryCamera, PhotogrammetryCameraSet],
        batch_size: int = 1,
        aggregate_img_scale: float = 1,
        check_null_image: bool = False,
        **pix2face_kwargs,
    ):
        """Find the per-face projection for each of a set of images and associated camera

        Args:
            cameras (typing.Union[PhotogrammetryCamera, PhotogrammetryCameraSet]):
                The cameras to project images from. cam.get_image() will be called on each one
            batch_size (int, optional):
                The number of cameras to compute correspondences for at once. Defaults to 1.
            aggregate_img_scale (float, optional):
                The scale of pixel-to-face correspondences image, as a fraction of the original
                image. Lower values lead to better runtimes but decreased precision at content
                boundaries in the images. Defaults to 1.
            check_null_image (bool, optional):
                Only do indexing if there are non-null image values. This adds additional overhead,
                but can save the expensive operation of indexing in cases where it would be a no-op.

        Yields:
            np.ndarray: The per-face projection of an image in the camera set
        """
        # Create a local mesh
        mesh = self.get_mesh_in_cameras_coords(cameras)

        n_faces = self.faces.shape[0]

        # Iterate over batch of the cameras
        batch_stop = max(len(cameras) - batch_size + 1, 1)
        for batch_start in range(0, batch_stop, batch_size):
            batch_inds = list(range(batch_start, batch_start + batch_size))
            batch_cameras = cameras.get_subset_cameras(batch_inds)
            # Compute a batch of pix2face correspondences. This is likely the slowest step
            batch_pix2face = self.pix2face(
                cameras=batch_cameras,
                mesh=mesh,
                render_img_scale=aggregate_img_scale,
                **pix2face_kwargs,
            )
            for i, pix2face in enumerate(batch_pix2face):
                img = cameras.get_image_by_index(batch_start + i, aggregate_img_scale)

                n_channels = 1 if img.ndim == 2 else img.shape[-1]
                textured_faces = np.full((n_faces, n_channels), fill_value=np.nan)

                # Only do the expensive indexing step if there are finite values in the image. This is most
                # significant for sparse detection tasks where some images may have no real data
                if not check_null_image or np.any(np.isfinite(img)):
                    flat_img = np.reshape(img, (img.shape[0] * img.shape[1], -1))
                    flat_pix2face = pix2face.flatten()
                    # TODO this creates ill-defined behavior if multiple pixels map to the same face
                    # my guess is the later pixel in the flattened array will override the former
                    # TODO make sure that null pix2face values are handled properly
                    textured_faces[flat_pix2face] = flat_img
                yield textured_faces

    def aggregate_projected_images(
        self,
        cameras: typing.Union[PhotogrammetryCamera, PhotogrammetryCameraSet],
        batch_size: int = 1,
        aggregate_img_scale: float = 1,
        return_all: bool = False,
        **kwargs,
    ):
        """Aggregate the imagery from multiple cameras into per-face averges

        Args:
            cameras (typing.Union[PhotogrammetryCamera, PhotogrammetryCameraSet]):
                The cameras to aggregate the images from. cam.get_image() will be called on each
                element.
            batch_size (int, optional):
                The number of cameras to compute correspondences for at once. Defaults to 1.
            aggregate_img_scale (float, optional):
                The scale of pixel-to-face correspondences image, as a fraction of the original
                image. Lower values lead to better runtimes but decreased precision at content
                boundaries in the images. Defaults to 1.
            return_all (bool, optional):
                Return the projection of each individual image, rather than just the aggregates.
                Defaults to False.

        Returns:
            np.ndarray: (n_faces, n_image_channels) The average projected image per face
            dict: Additional information, including the summed projections, observations per face,
                  and potentially each individual projection
        """
        project_images_generator = self.project_images(
            cameras=cameras,
            batch_size=batch_size,
            aggregate_img_scale=aggregate_img_scale,
            **kwargs,
        )

        if return_all:
            all_projections = []

        # TODO this should be a convenience method
        n_faces = self.faces.shape[0]

        projection_counts = np.zeros(n_faces)
        summed_projection = None

        for projection_for_image in tqdm(
            project_images_generator,
            total=len(cameras),
            desc="Aggregating projected viewpoints",
        ):
            if return_all:
                all_projections.append(projection_for_image)

            if summed_projection is None:
                summed_projection = projection_for_image.astype(float)
            else:
                summed_projection = np.nansum(
                    [summed_projection, projection_for_image], axis=0
                )

            projected_faces = np.any(np.isfinite(projection_for_image), axis=1).astype(
                int
            )
            projection_counts += projected_faces

        no_projections = projection_counts == 0
        summed_projection[no_projections] = np.nan

        additional_information = {
            "projection_counts": projection_counts,
            "summed_projections": summed_projection,
        }

        if return_all:
            additional_information["all_projections"] = all_projections

        average_projections = np.divide(
            summed_projection, np.expand_dims(projection_counts, 1)
        )

        return average_projections, additional_information

    # Visualization and saving methods
    def vis(
        self,
        plotter: pv.Plotter = None,
        interactive: bool = True,
        camera_set: PhotogrammetryCameraSet = None,
        screenshot_filename: PATH_TYPE = None,
        vis_scalars: typing.Union[None, np.ndarray] = None,
        mesh_kwargs: typing.Dict = None,
        interactive_jupyter: bool = False,
        plotter_kwargs: typing.Dict = {},
        enable_ssao: bool = True,
        force_xvfb: bool = False,
        frustum_scale: float = 2,
        IDs_to_labels: typing.Union[None, dict] = None,
    ):
        """Show the mesh and cameras

        Args:
            plotter (pyvista.Plotter, optional):
                Plotter to use, else one will be created
            off_screen (bool, optional):
                Show offscreen
            camera_set (PhotogrammetryCameraSet, optional):
                Cameras to visualize. Defaults to None.
            screenshot_filename (PATH_TYPE, optional):
                Filepath to save to, will show interactively if None. Defaults to None.
            vis_scalars (None, np.ndarray):
                Scalars to show
            mesh_kwargs:
                dict of keyword arguments for the mesh
            interactive_jupyter (bool):
                Should jupyter windows be interactive. This doesn't always work, especially on VSCode.
            plotter_kwargs:
                dict of keyword arguments for the plotter
            frustum_scale (float, optional):
                Size of cameras in world units. Defaults to None.
            IDs_to_labels ([None, dict], optional):
                Mapping from IDs to human readable labels for discrete classes. Defaults to the mesh
                IDs_to_labels if unset.
        """
        # TODO conside reprojecting to ensure axes are both meters-based
        off_screen = (not interactive) or (screenshot_filename is not None)

        # If the IDs to labels is not set, use the default ones for this mesh
        if IDs_to_labels is None:
            IDs_to_labels = self.get_IDs_to_labels()

        # Set the mesh kwargs if not set
        if mesh_kwargs is None:
            # This needs to be a dict, even if it's empty
            mesh_kwargs = {}

            # If there are discrete labels, set the colormap and limits inteligently
            if IDs_to_labels is not None:
                # Compute the largest ID
                max_ID = max(IDs_to_labels.keys())
                if max_ID < 20:
                    colors = [
                        matplotlib.colors.to_hex(c)
                        for c in plt.get_cmap(
                            ("tab10" if max_ID < 10 else "tab20")
                        ).colors
                    ]
                    mesh_kwargs["cmap"] = colors[0 : max_ID + 1]
                    mesh_kwargs["clim"] = (-0.5, max_ID + 0.5)

        # Create the plotter if it's None
        plotter = create_pv_plotter(
            off_screen=off_screen, force_xvfb=force_xvfb, plotter=plotter
        )

        # If the vis scalars are None, use the saved texture
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
            )

        is_rgb = (
            self.pyvista_mesh.active_scalars_name == "RGB"
            if vis_scalars is None
            else (vis_scalars.ndim == 2 and vis_scalars.shape[1] > 1)
        )

        # Data in the range [0, 255] must be uint8 type
        if is_rgb and np.nanmax(vis_scalars) > 1.0:
            vis_scalars = np.clip(vis_scalars, 0, 255).astype(np.uint8)

        scalar_bar_args = {"vertical": True}
        if IDs_to_labels is not None and "annotations" not in mesh_kwargs:
            mesh_kwargs["annotations"] = IDs_to_labels
            scalar_bar_args["n_labels"] = 0

        vis_mesh = self.reproject_CRS(EARTH_CENTERED_EARTH_FIXED_CRS, inplace=False)

        # If camera set is provided, transform the mesh into those coordinates
        if camera_set is not None:
            # Compute the transform mapping from the earth centered, earth fixed coordinate frame
            # (EPSG:4978) to the coordinate of the camera
            epsg_4978_to_camera = np.linalg.inv(
                camera_set.get_local_to_epsg_4978_transform()
            )
            # Apply the 4x4 transform using the pyvista transform method to get the mesh into the
            # same coordinate frame as the cameras.
            vis_mesh.transform(epsg_4978_to_camera, inplace=True)

        # Add the mesh
        plotter.add_mesh(
            vis_mesh,
            scalars=vis_scalars,
            rgb=is_rgb,
            scalar_bar_args=scalar_bar_args,
            **mesh_kwargs,
        )
        # If the camera set is provided, show this too
        if camera_set is not None:
            # Adjust the frustum scale if the mesh came from metashape
            # Find the cube root of the determinant of the upper-left 3x3 submatrix to find the scaling factor
            if (
                camera_set.get_local_to_epsg_4978_transform() is not None
                and frustum_scale is not None
            ):
                transform_determinant = np.linalg.det(
                    camera_set.get_local_to_epsg_4978_transform()[:3, :3]
                )
                scale_factor = np.cbrt(transform_determinant)
                frustum_scale = frustum_scale / scale_factor
            camera_set.vis(
                plotter, add_orientation_cube=False, frustum_scale=frustum_scale
            )

        # Enable screen space shading
        if enable_ssao:
            plotter.enable_ssao()

        # Create parent folder if none exists
        if screenshot_filename is not None:
            ensure_containing_folder(screenshot_filename)

        if "jupyter_backend" not in plotter_kwargs:
            if interactive_jupyter:
                plotter_kwargs["jupyter_backend"] = "trame"
            else:
                plotter_kwargs["jupyter_backend"] = "static"

        if "title" not in plotter_kwargs:
            plotter_kwargs["title"] = "Geograypher mesh viewer"

        # Show
        return plotter.show(
            screenshot=screenshot_filename,
            **plotter_kwargs,
        )

    def save_renders(
        self,
        camera_set: PhotogrammetryCameraSet,
        render_image_scale=1.0,
        output_folder: PATH_TYPE = Path(VIS_FOLDER, "renders"),
        make_composites: bool = False,
        save_native_resolution: bool = False,
        cast_to_uint8: bool = True,
        save_as_npy: bool = False,
        uint8_value_for_null_texture: np.uint8 = NULL_TEXTURE_INT_VALUE,
        **render_kwargs,
    ):
        """Render an image from the viewpoint of each specified camera and save a composite

        Args:
            camera_set (PhotogrammetryCameraSet):
                Camera set to use for rendering
            render_image_scale (float, optional):
                Multiplier on the real image scale to obtain size for rendering. Lower values
                yield a lower-resolution render but the runtime is quiker. Defaults to 1.0.
            render_folder (PATH_TYPE, optional):
                Save images to this folder. Defaults to Path(VIS_FOLDER, "renders")
            make_composites (bool, optional):
                Should a triple pane composite with the original image be saved rather than the
                raw label
            cast_to_uint8: (bool, optional):
                cast the float valued data to unit8 for saving efficiency. May dramatically increase
                efficiency due to tif compression. Saves as tif unless save_as_npy is specified as True.
            save_as_npy (bool, optional):
                Save the rendered images as numpy arrays rather than TIF images. Defaults to False.
            uint8_value_for_null_texture (np.uint8, optional):
                What value to assign for values that can't be represented as unsigned 8-bit data.
                Defaults to NULL_TEXTURE_INT_VALUE
            render_kwargs:
                keyword arguments passed to the render.
        """

        ensure_folder(output_folder)
        self.logger.info(f"Saving renders to {output_folder}")

        # Save the classes filename
        self.save_IDs_to_labels(Path(output_folder, "IDs_to_labels.json"))

        # Create the generator object to render the images
        # Since this is a generator, this will be fast
        render_gen = self.render_flat(
            camera_set,
            render_img_scale=render_image_scale,
            return_camera=True,
            distortion_set=camera_set,
            **render_kwargs,
        )

        # The computation only happens when items are requested from the generator
        for rendered, camera in tqdm(
            render_gen,
            total=len(camera_set),
            desc="Computing and saving renders",
        ):
            ## All this is post-processing to visualize the rendered label.
            # rendered could either be a one channel image of integer IDs,
            # a one-channel image of scalars, or a three-channel image of
            # RGB. It could also be multi-channel image corresponding to anything,
            # but we don't expect that yet
            if save_native_resolution and render_image_scale != 1:
                native_size = camera.get_image_size()
                # Upsample using nearest neighbor interpolation for discrete labels and
                # bilinear for non-discrete
                # TODO this will need to be fixed for multi-channel images since I don't think resize works
                rendered = resize(
                    rendered,
                    native_size,
                    order=(0 if self.is_discrete_texture() else 1),
                )

            if cast_to_uint8:
                # Deterimine values that cannot be represented as uint8
                mask = np.logical_or.reduce(
                    [
                        rendered < 0,
                        rendered > 255,
                        np.logical_not(np.isfinite(rendered)),
                    ]
                )
                rendered[mask] = uint8_value_for_null_texture
                # Cast and squeeze since you can't save a one-channel image
                rendered = np.squeeze(rendered.astype(np.uint8))

            if make_composites:
                RGB_image = camera.get_image(
                    image_scale=(1.0 if save_native_resolution else render_image_scale)
                )
                rendered = create_composite(
                    RGB_image=RGB_image,
                    label_image=rendered,
                    IDs_to_labels=self.get_IDs_to_labels(),
                )
            else:
                # Clip channels if needed
                if rendered.ndim == 3:
                    rendered = rendered[..., :3]

            try:
                # If the filename stored with the camera data [camera.get_image_filename]
                # is a subpath of your camera set image folder, use the same subpath for
                # the output data.
                camera_filename = camera.get_image_filename().relative_to(
                    camera_set.image_folder
                )
            except ValueError:
                raise ValueError(
                    "Tried to find the relative path of the camera path"
                    f" ({camera.get_image_filename()}) inside of the camera set image"
                    f" folder ({camera_set.image_folder}), but failed. The tool being called"
                    " may have an 'original_image_folder' argument, which could be used to"
                    " delete the initial, mismatched portion of the camera path. See more here:"
                    " https://github.com/open-forest-observatory/automate-metashape/issues/90."
                )
            output_filename = Path(output_folder, camera_filename)

            # This may create nested folders in the output dir
            ensure_containing_folder(output_filename)

            if save_as_npy is True:
                output_filename = str(output_filename.with_suffix(".npy"))
                # Save the image
                np.save(output_filename, rendered)
            else:
                # Save image as TIF
                output_filename = str(output_filename.with_suffix(".tif"))
                # Remove singleton channel dimension (1, H, W) -> (H, W) to save single-channel TIF
                rendered = np.squeeze(rendered)
                # TODO: Consider supporting TIF files with float data (like CHM renders) by adding a separate flag.
                # Evaluate whether this offers more space savings than npy files.
                # If cast_to_uint8 is True, rendered is already in uint8
                if cast_to_uint8 is False:
                    # Check if max value in the rendered image is within the range of uint16
                    if np.nanmax(rendered) <= np.iinfo(np.uint16).max:
                        # Cast from float to uint16
                        rendered = rendered.astype(np.uint16)
                    else:
                        rendered = rendered.astype(np.uint32)

                # Save the image
                skimage.io.imsave(
                    output_filename,
                    rendered,
                    compression="deflate",
                    check_contrast=False,
                )

    def export_covering_meshes(
        self,
        N: int,
        z_buffer: tuple = (0, 0),
        subsample: typing.Union[int, None] = None,
    ) -> typing.Tuple[pv.PolyData, pv.PolyData]:
        """
        This function will process self.pyvista_mesh. It will start by identifying the (x, y)
        boundaries of that mesh, then create an (N, N) grid of (x, y) points over that area.
        At each (x, y) square in the grid, the pyvista mesh will be sampled for the highest and
        lowest value. This will be used to set the Z heights for that grid point (plus the
        z_buffer). Then upper and lower bound surfaces will be made from these grid points
        using delaunay_2d and returned.

        Args:
            N (int): Number of sample points to take as a grid
            z_buffer (tuple): Offset in Z to give the sampled points, in the units of the mesh.
                [0] is for the upper mesh, [1] is for the lower.
            subsample (int / None): If not None, we will naively subsample self.pyvista_mesh.points
                to speed up runtime by [::subsample]. A larger number will subsample more.

        Returns:
            tuple: A tuple of two pyvista.PolyData objects, the first is the upper
            covering mesh, the second the lower covering mesh.
        """

        assert self.pyvista_mesh is not None, "Requires a populated mesh"
        assert len(z_buffer) == 2, "2 buffers (top, bottom) are required"

        # Get mesh points
        points = self.pyvista_mesh.points
        if len(points) == 0:
            return (self.pyvista_mesh.copy(), self.pyvista_mesh.copy())

        if subsample is not None:
            points = points[::subsample]
        x_min, y_min = points[:, 0].min(), points[:, 1].min()
        x_max, y_max = points[:, 0].max(), points[:, 1].max()

        # Create grid
        grid_ind = np.indices((N, N)).reshape(2, -1).T
        x_grid = np.linspace(x_min, x_max, N)
        y_grid = np.linspace(y_min, y_max, N)
        grid_points = np.column_stack([x_grid[grid_ind[:, 0]], y_grid[grid_ind[:, 1]]])

        # For each grid point, find mesh points within the cell and get z value
        cell_w_half = (x_max - x_min) / (N - 1) / 2
        cell_h_half = (y_max - y_min) / (N - 1) / 2
        # Positive
        z_p_values = np.full(grid_points.shape[0], np.nan)
        # Negative
        z_n_values = np.full(grid_points.shape[0], np.nan)

        # Precompute mask stripes to save time
        x_masks = {
            index: (points[:, 0] >= x_grid[index] - cell_w_half)
            & (points[:, 0] <= x_grid[index] + cell_w_half)
            for index in range(N)
        }
        y_masks = {
            index: (points[:, 1] >= y_grid[index] - cell_h_half)
            & (points[:, 1] <= y_grid[index] + cell_h_half)
            for index in range(N)
        }
        for i, (xi, yi) in enumerate(tqdm(grid_ind, desc="Building covering meshes")):
            # Find mesh points within the cell
            mask = x_masks[xi] & y_masks[yi]
            z_candidates = points[mask, 2]
            if len(z_candidates) > 0:
                z_p_values[i] = np.max(z_candidates) + z_buffer[0]
                z_n_values[i] = np.min(z_candidates) + z_buffer[1]

        # Create 3D points for the grid
        grid_p_3d = np.column_stack([grid_points, z_p_values])
        grid_n_3d = np.column_stack([grid_points, z_n_values])
        # Remove points with no Z values
        grid_p_3d = grid_p_3d[~np.isnan(z_p_values)]
        grid_n_3d = grid_n_3d[~np.isnan(z_n_values)]

        # Create pyvista point cloud and surface
        return (
            pv.PolyData(grid_p_3d).delaunay_2d(),
            pv.PolyData(grid_n_3d).delaunay_2d(),
        )
