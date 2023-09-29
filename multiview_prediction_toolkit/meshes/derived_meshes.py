import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio as rio
import torch
from pytorch3d.renderer import TexturesVertex
from pytorch3d.structures import Meshes
from scipy.spatial.distance import cdist
from shapely import Point

from multiview_prediction_toolkit.config import (
    COLORS,
    DEFAULT_DEM_FILE,
    DEFAULT_GEOPOLYGON_FILE,
    PATH_TYPE,
)
from multiview_prediction_toolkit.meshes.meshes import TexturedPhotogrammetryMesh


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
        self.color_with_binary_mask(
            ground_points,
            color_true=np.array(COLORS["earth"]) / 255.0,
            color_false=np.array(COLORS["canopy"]) / 255.0,
        )

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
            is_tree = inside_tree_polygon.to(self.device).to(torch.bool)

        self.texture_with_binary_mask(
            is_tree,
            color_true=np.array(COLORS["canopy"]) / 255.0,
            color_false=np.array(COLORS["earth"]) / 255.0,
            vis=vis,
        )

    def create_texture(
        self,
        geo_polygon_file: PATH_TYPE = DEFAULT_GEOPOLYGON_FILE,
        DEM_file: PATH_TYPE = DEFAULT_DEM_FILE,
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
        else:
            self.create_texture_height_threshold(
                DEM_file=DEM_file,
                ground_height_threshold=ground_height_threshold,
                vis=vis,
            )
