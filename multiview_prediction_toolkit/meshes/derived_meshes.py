import geopandas as gpd
from matplotlib import legend
from networkx import union
import numpy as np
import pandas as pd
import rasterio as rio
import torch
import matplotlib.pyplot as plt
from pytorch3d.renderer import TexturesVertex
from pytorch3d.structures import Meshes
from scipy.spatial.distance import cdist
from shapely import Point, MultiPolygon
from shapely.validation import make_valid
from shapely import difference
from geopandas import GeoDataFrame


from multiview_prediction_toolkit.config import (
    COLORS,
    DEFAULT_DEM_FILE,
    DEFAULT_GEOPOLYGON_FILE,
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
        textures = TexturesVertex(verts_features=verts_rgb).to(self.device)
        self.pytorch_mesh = Meshes(
            verts=[self.verts], faces=[self.faces], textures=textures
        )
        self.pytorch_mesh = self.pytorch_mesh.to(self.device)


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


class GeodataPhotogrammetryMesh(TexturedPhotogrammetryMesh):
    def get_verts_geodataframe(self, crs, east_is_first=True):
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
        df["coords"] = list(zip(df["east"], df["north"]))
        df["coords"] = df["coords"].apply(Point)
        points = gpd.GeoDataFrame(df, geometry="coords", crs=crs)

        # Add an index column because the normal index will not be preserved in future operations
        points["id"] = df.index

        return points

    def get_values_from_geopandas(
        self, geopandas_df: GeoDataFrame, geopandas_column: str
    ):
        points = self.get_verts_geodataframe(geopandas_df.crs)

        # Select points that are within the polygons
        points_in_polygons = gpd.tools.overlay(points, geopandas_df, how="intersection")
        # Create an array corresponding to all the points and initialize to NaN
        values = np.full(shape=points.shape[0], fill_value=np.nan)
        # Assign points that are inside a given tree with that tree's ID
        values[points_in_polygons["id"].to_numpy()] = points_in_polygons[
            geopandas_column
        ].to_numpy()
        return values

    def get_height_above_ground(self, DEM_file: PATH_TYPE = DEFAULT_DEM_FILE):
        """Compute the height above groun for each point on the mesh

        Args:
            DEM_file (PATH_TYPE, optional): The path the the DEM/DTM file from metashape. Defaults to DEFAULT_DEM_FILE.

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

    def create_texture_height_threshold(
        self,
        DEM_file: PATH_TYPE = DEFAULT_DEM_FILE,
        ground_height_threshold=2,
        vis: bool = False,
    ):
        """Texture by thresholding the height above groun

        Args:
            DEM_file (PATH_TYPE, optional): Filepath for DEM/DTM file from metashape. Defaults to DEFAULT_DEM_FILE.
            threshold (int, optional): Height above gound to be considered not ground (meters). Defaults to 2.
        """
        # Get the height of each mesh point above the ground
        height_above_ground = self.get_height_above_ground(DEM_file=DEM_file)
        # Threshold to dermine if it's ground or not
        ground_points = height_above_ground < ground_height_threshold
        # Color the mesh with this mask
        self.texture_with_binary_mask(
            ground_points,
            color_true=np.array(COLORS["earth"]) / 255.0,
            color_false=np.array(COLORS["canopy"]) / 255.0,
        )

    def create_texture_geopoints(
        self,
        geopoints_file: PATH_TYPE,
        radius_meters: float = 3,
        vis: bool = True,
        ground_height_threshold: float = 2,
    ):
        """Creates a classification texture from a circle around each point in a file

        Args:
            geopoints_file (PATH_TYPE): Path to a geofile that can be read by geopandas
            radius_meters (float, optional): _description_. Defaults to 1.
            vis (bool, optional): _description_. Defaults to True.
        """
        # Read in the data
        geopoints = gpd.read_file(geopoints_file)
        # Determine the projective CRS for the region since many operation don't work on geographic CRS
        first_point = geopoints["geometry"][0]
        projected_CRS = get_projected_CRS(first_point.y, first_point.x)

        # transfrom to a projective CRS
        geopoints = geopoints.to_crs(crs=projected_CRS)
        # Get the size of the trees
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

        # Now create circles around each point
        geopolygons = geopoints
        geopolygons["geometry"] = geopoints["geometry"].buffer(radius)

        # Split
        split_by_species = list(geopolygons.groupby("Species", axis=0))
        species_multipolygon_dict = {
            species_ID: make_valid(species_df.unary_union)
            for species_ID, species_df in split_by_species
        }
        # union_of_intrsections = find_union_of_intersections(
        #    list(species_multipolygon_dict.values()), crs=projected_CRS
        # )
        # species_multipolygon_dict = {
        #    species_ID: difference(species_multipolygon, union_of_intrsections)
        #    for species_ID, species_multipolygon in species_multipolygon_dict.items()
        # }
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

        species_int_IDs = np.load("vis/species_int_IDs.npy")
        species_int_IDs = np.nan_to_num(species_int_IDs, nan=-1).astype(int)

        ground_points = self.get_height_above_ground() < ground_height_threshold
        species_int_IDs[ground_points] = -1

        self.vertex_IDs = species_int_IDs

        self.pytorch_mesh = Meshes(
            verts=[torch.Tensor(self.verts).to(self.device)],
            faces=[torch.Tensor(self.faces).to(self.device)],
        )
        self.pytorch_mesh = self.pytorch_mesh.to(self.device)

    def create_texture_geopolygon(
        self,
        geo_polygon_file: PATH_TYPE = None,
        DEM_file: PATH_TYPE = None,
        ground_height_threshold=2,
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

        tree_IDs = self.get_values_from_geopandas(geo_polygons, "treeID")

        # These points are within a tree
        # TODO, in the future we might want to do something more sophisticated than tree/not tree
        inside_tree_polygon = np.isfinite(tree_IDs)

        if DEM_file is not None:
            above_ground = (
                self.get_height_above_ground(DEM_file=DEM_file)
                > ground_height_threshold
            )
            is_tree = (
                torch.Tensor(np.logical_and(inside_tree_polygon, above_ground))
                .to(self.device)
                .to(torch.bool)
            )
        else:
            is_tree = torch.Tensor(inside_tree_polygon).to(self.device).to(torch.bool)

        self.vert_to_face_IDs(is_tree.cpu().numpy())

        self.texture_with_binary_mask(
            is_tree,
            color_true=np.array(COLORS["canopy"]) / 255.0,
            color_false=np.array(COLORS["earth"]) / 255.0,
            vis=vis,
        )

    def create_texture(
        self,
        geo_polygon_file: PATH_TYPE = None,
        geo_point_file: PATH_TYPE = None,
        DEM_file: PATH_TYPE = None,
        ground_height_threshold=2,
        vis: bool = False,
    ):
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
