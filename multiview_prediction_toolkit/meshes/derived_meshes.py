import geopandas as gpd
import pyproj
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from pytorch3d.renderer import TexturesVertex
from pytorch3d.structures import Meshes
from scipy.spatial.distance import cdist
from shapely import Point, difference
from shapely.validation import make_valid
from geopandas import GeoDataFrame


from multiview_prediction_toolkit.config import (
    COLORS,
    DEFAULT_DEM_FILE,
    PATH_TYPE,
)
from multiview_prediction_toolkit.meshes.meshes import TexturedPhotogrammetryMesh
from multiview_prediction_toolkit.utils.utils import (
    find_union_of_intersections,
    get_projected_CRS,
    to_float,
)


class ColorPhotogrammetryMesh(TexturedPhotogrammetryMesh):
    def create_texture(self):
        # Create a texture from the colors
        # Convert RGB values to [0,1] and format correctly
        if "RGB" in self.pyvista_mesh.array_names:
            verts_rgb = torch.Tensor(
                np.expand_dims(self.pyvista_mesh["RGB"] / 255, axis=0)
            ).to(self.device)
        else:
            # Default gray color
            # TODO warn that there was not texture
            # TODO long-term, query the orignal vertices and get color of closest one
            print(self.pyvista_mesh.array_names)
            verts_rgb = torch.Tensor(
                np.full((1, self.pyvista_mesh.n_points, 3), 0.5)
            ).to(self.device)
        self.create_pytorch_3d_mesh(vert_texture=verts_rgb)


class DummyPhotogrammetryMesh(TexturedPhotogrammetryMesh):
    def create_texture(self, use_colorseg: bool = True):
        """Create a dummy texture for debuging

        Args:
            use_colorseg (bool, optional):
                Segment based on color into two classes
                Otherwise, segment based on a centered circle. Defaults to True.
        """
        CANOPY_COLOR = COLORS["canopy"]
        EARTH_COLOR = COLORS["earth"]
        RGB_values = self.pyvista_mesh["RGB"]
        if use_colorseg:
            test_values = np.array([CANOPY_COLOR, EARTH_COLOR])
            dists = cdist(RGB_values, test_values)
            mask = np.argmin(dists, axis=1).astype(bool)
        else:
            XYZ_values = self.pyvista_mesh.points
            center = np.mean(XYZ_values, axis=0, keepdims=True)
            dists_to_center = np.linalg.norm(XYZ_values[:, :2] - center[:, :2], axis=1)
            cutoff_value = np.quantile(dists_to_center, [0.1])
            mask = dists_to_center > cutoff_value

        self.texture_with_binary_mask(
            mask, color_true=EARTH_COLOR, color_false=CANOPY_COLOR
        )


class HeightAboveGroundPhotogrammertryMesh(TexturedPhotogrammetryMesh):
    def __init__(
        self,
        mesh_filename: PATH_TYPE,
        downsample_target: float = 1,
        use_pytorch3d_mesh: bool = True,
        DEM_file: PATH_TYPE = DEFAULT_DEM_FILE,
        ground_height_threshold=2,
        **kwargs
    ):
        """Texture by thresholding the height above groun

        Args:
            DEM_file (PATH_TYPE, optional): Filepath for DEM/DTM file from metashape. Defaults to DEFAULT_DEM_FILE.
            threshold (int, optional): Height above gound to be considered not ground (meters). Defaults to 2.
        """
        super().__init__(mesh_filename, downsample_target, use_pytorch3d_mesh=False)
        self.use_pytorch3d_mesh = use_pytorch3d_mesh

        # Get the height of each mesh point above the ground
        self.vertex_IDs = self.get_height_above_ground(
            DEM_file=DEM_file, threshold=ground_height_threshold
        ).astype(int)

        self.create_pytorch_3d_mesh()


class TreeSpeciesTexturedPhotogrammetryMesh(TexturedPhotogrammetryMesh):
    def __init__(
        self,
        mesh_filename: PATH_TYPE,
        geopoints_file: PATH_TYPE,
        downsample_target: float = 1,
        texture: np.ndarray = None,
        use_pytorch3d_mesh: bool = True,
        radius_meters: float = 3,
        vis: bool = True,
        ground_height_threshold: float = 2,
        DEM_file: PATH_TYPE = DEFAULT_DEM_FILE,
        discard_overlapping: bool = False,
    ):
        """Creates a classification texture from a circle around each point in a file

        Args:
            geopoints_file (PATH_TYPE): Path to a geofile that can be read by geopandas
            radius_meters (float, optional): Radius around each point to use. Defaults to 1.
            vis (bool, optional): Visualize. Defaults to True.
            ground_height_threshold (float, optional): Points under this height are considered ground
            DEM_file (PATH_TYPE, optional): The DEM file to use
            discard_overlapping (bool, optional): Discard regions where two classes disagree
        """
        super().__init__(mesh_filename, downsample_target, texture, use_pytorch3d_mesh)
        # Read in the data
        geopoints = gpd.read_file(geopoints_file)
        # Determine the projective CRS for the region since many operation don't work on geographic CRS
        first_point = geopoints["geometry"][0]
        projected_CRS = get_projected_CRS(first_point.y, first_point.x)

        # transfrom to a projective CRS
        geopoints = geopoints.to_crs(crs=projected_CRS)
        # Get the size of the trees
        radius = self.get_radius(geopoints=geopoints, radius=radius)

        # Now create circles around each point
        geopolygons = geopoints
        geopolygons["geometry"] = geopoints["geometry"].buffer(radius)

        # Split
        split_by_species = list(geopolygons.groupby("Species", axis=0))
        species_multipolygon_dict = {
            species_ID: make_valid(species_df.unary_union)
            for species_ID, species_df in split_by_species
        }
        # Should points regions with multiple different species be discarded?
        if discard_overlapping:
            union_of_intrsections = find_union_of_intersections(
                list(species_multipolygon_dict.values()), crs=projected_CRS
            )
            species_multipolygon_dict = {
                species_ID: difference(species_multipolygon, union_of_intrsections)
                for species_ID, species_multipolygon in species_multipolygon_dict.items()
            }
        else:
            species_multipolygon_dict = {
                species_ID: species_multipolygon
                for species_ID, species_multipolygon in species_multipolygon_dict.items()
            }
        geopandas_all_species = GeoDataFrame(
            geometry=list(species_multipolygon_dict.values()), crs=projected_CRS
        )
        geopandas_all_species["species"] = list(species_multipolygon_dict.keys())
        # The normal index won't be preserved in future operations
        # This might be faster than returning a string type?
        geopandas_all_species["species_int_ID"] = geopandas_all_species.index
        if vis:
            geopandas_all_species.plot("species", legend=True)
            plt.show()

        species_int_IDs = self.get_values_for_verts_from_vector(
            column_names="species_int_ID", geopandas_df=geopandas_all_species
        )

        ground_points = self.get_height_above_ground(
            DEM_file, threshold=ground_height_threshold
        )
        species_int_IDs[ground_points] = -1

        self.vertex_IDs = species_int_IDs

    def get_radius(self, geopoints, radius_meters):
        crown_width_1 = geopoints["Crown_width_1"].to_numpy()
        crown_width_2 = geopoints["Crown_width_2"].to_numpy()
        crown_width_1 = np.array([to_float(x, "NA") for x in crown_width_1])
        crown_width_2 = np.array([to_float(x, "NA") for x in crown_width_2])
        null_value = 1000
        min_width = np.ones_like(crown_width_1) * null_value
        radius = np.nanmin(
            np.vstack((crown_width_1, crown_width_2, min_width)).T, axis=1
        )
        null_entries = radius == null_value
        radius = radius * 0.75
        radius = np.clip(radius, 0, 20)
        radius[null_entries] = radius_meters

        return radius


class GeodataPhotogrammetryMesh(TexturedPhotogrammetryMesh):
    def __init__(
        self,
        mesh_filename: PATH_TYPE,
        downsample_target: float = 1,
        geo_polygon_file: PATH_TYPE = None,
        geo_point_file: PATH_TYPE = None,
        DEM_file: PATH_TYPE = None,
        ground_height_threshold=2,
        vis: bool = False,
        **kwargs
    ):
        # Load and downsample the mesh
        super().__init__(
            mesh_filename=mesh_filename,
            downsample_target=downsample_target,
            use_pytorch3d_mesh=False,
        )

        # Add a texture
        if geo_polygon_file is not None:
            self.create_texture_geopolygon(
                geo_polygon_file=geo_polygon_file,
                DEM_file=DEM_file,
                ground_height_threshold=ground_height_threshold,
                vis=vis,
            )
        elif geo_point_file is not None:
            self.create_texture_geopoints(geopoints_file=geo_point_file)
        else:
            self.create_texture_height_threshold(
                DEM_file=DEM_file,
                ground_height_threshold=ground_height_threshold,
                vis=vis,
            )

    def create_texture_geopolygon(
        self,
        geo_polygon_file: PATH_TYPE = None,
        DEM_file: PATH_TYPE = None,
        ground_height_threshold=2,
    ):
        """Create a texture from a geofile containing polygons

        Args:
            geo_polygon_file (PATH_TYPE, optional):
                Filepath to read from. Must be able to be opened by geopandas. Defaults to DEFAULT_GEOPOLYGON_FILE.
            vis (bool, optional): Show the texture. Defaults to False.
        """
        # Get the tree IDs from the file
        tree_IDs = self.get_values_for_verts_from_vector(
            column_names="treeID", vector_file=geo_polygon_file
        )

        # These points are within a tree
        # TODO, in the future we might want to do something more sophisticated than tree/not tree
        is_tree = np.isfinite(tree_IDs)

        if DEM_file is not None:
            above_ground = np.logical_not(
                self.get_height_above_ground(
                    DEM_file=DEM_file, threshold=ground_height_threshold
                )
            )
            is_tree = np.logical_and(is_tree, above_ground)

        self.vertex_IDs = is_tree
