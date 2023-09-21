import geopandas as gpd
import pandas as pd
import pyvista as pv
import numpy as np
import time
from shapely.geometry import Point


# The location of the files we need
GEO_DATA_FILE = "/ofo-share/repos-david/semantic-mesh-pytorch3d/data/composite_20230520T0519/composite_20230520T0519_crowns.gpkg"
GEO_MESH = "/ofo-share/repos-david/semantic-mesh-pytorch3d/data/composite_georef/composite_georef.obj"
LOCAL_MESH = "/ofo-share/repos-david/semantic-mesh-pytorch3d/data/composite_georef/composite_local_reset.obj"

# Read the data
gdf = gpd.read_file(GEO_DATA_FILE)
# convert to lat, lon so it matches the mesh
gdf = gdf.to_crs("EPSG:4326")

# Load the mesh, assumed to be lat, lon
g_mesh = pv.read(GEO_MESH)
# Note that lat, lon convention doesn't correspond to how it's said
lat = np.array(g_mesh.points[:,1])
lon = np.array(g_mesh.points[:,0])
# Normalize this axis since it's in meters and the rest are lat-lon
g_mesh.points[:,2] = g_mesh.points[:,2] / 111119

# Taken from https://www.matecdev.com/posts/point-in-polygon.html
df = pd.DataFrame({'lon':lon, 'lat':lat})
df['coords'] = list(zip(df['lon'],df['lat']))
df['coords'] = df['coords'].apply(Point)
points = gpd.GeoDataFrame(df, geometry='coords', crs=gdf.crs)
points["id"] = df.index
print(f"About to do join with {points.shape[0]} points")
start = time.time()
pointInPolyws = gpd.tools.overlay(points, gdf, how='intersection')
print(f"Finished overlay in {time.time() - start} seconds")
polygon_IDs = np.full(shape=points.shape[0], fill_value=np.nan)
polygon_IDs[pointInPolyws["id"].to_numpy()] = pointInPolyws["treeID"].to_numpy()
print(np.unique(polygon_IDs))

g_mesh["is_tree"] = np.isfinite(polygon_IDs)
local_mesh = pv.read(LOCAL_MESH)
local_mesh["is_tree"] = g_mesh["is_tree"]
local_mesh.plot()
#g_mesh.plot()
#g_point_cloud = pv.PolyData(g_mesh.points)
#g_point_cloud["is_tree"] = g_mesh["is_tree"]
#g_point_cloud.plot()

