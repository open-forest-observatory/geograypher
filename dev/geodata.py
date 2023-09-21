import geopandas as gpd
import pyvista as pv
import matplotlib.pyplot as plt
import shapely
import numpy as np
from tqdm import tqdm

# The location of the files we need
GEO_DATA_FILE = "/ofo-share/repos-david/semantic-mesh-pytorch3d/data/composite_20230520T0519/composite_20230520T0519_crowns.gpkg"
GEO_POINTCLOUD = "/ofo-share/repos-david/semantic-mesh-pytorch3d/data/composite_georef/composite_georef.obj"

# Read the data
gdf = gpd.read_file(GEO_DATA_FILE)
# convert to lat, lon so it matches the mesh
gdf = gdf.to_crs("EPSG:4326")
goem_array = gdf.geometry.tolist()
multipolygon = shapely.MultiPolygon(goem_array)
multipolygon = shapely.make_valid(multipolygon)
print(multipolygon.contains(shapely.Point(752325, 4317104)))
p = gpd.GeoSeries(multipolygon)


# https://www.matecdev.com/posts/point-in-polygon.html

g_point_cloud = pv.read(GEO_POINTCLOUD)
points = g_point_cloud.points
random_selection = np.random.choice(points.shape[0], int(1e6))
subsampled_points = points[random_selection]
subsampled_points[:, 2] = subsampled_points[:, 2] / 111119

insides = []
for point in tqdm(subsampled_points):
    insides.append(multipolygon.contains(shapely.Point(point[0], point[1])))

    pass

subsampled_pointcloud = pv.PolyData(subsampled_points)
subsampled_pointcloud.plot()
g_point_cloud.plot()
