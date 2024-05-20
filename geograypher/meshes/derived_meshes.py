import typing

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pyproj
from shapely import Point
from sklearn.cluster import KMeans
from tqdm import tqdm

from geograypher.cameras import PhotogrammetryCamera, PhotogrammetryCameraSet
from geograypher.constants import CACHE_FOLDER, PATH_TYPE
from geograypher.meshes import TexturedPhotogrammetryMesh
from geograypher.utils.geospatial import ensure_geometric_CRS


class TexturedPhotogrammetryMeshChunked(TexturedPhotogrammetryMesh):
    def get_mesh_chunks_for_cameras(
        self,
        cameras: typing.Union[PhotogrammetryCamera, PhotogrammetryCameraSet],
        n_clusters: int = 8,
        buffer_dist_meters=50,
        vis_clusters: bool = False,
    ):
        """
        Return the chunked meshes as generator
        """
        # Extract the points depending on whether it's a single camera or a set
        if isinstance(cameras, PhotogrammetryCamera):
            camera_points = [Point(*cameras.get_lon_lat())]
        else:
            # Get the lat lon for each camera point and turn into a shapely Point
            camera_points = [Point(*cp) for cp in cameras.get_lon_lat_coords()]

        # Create a geodataframe from the points
        camera_points = gpd.GeoDataFrame(
            geometry=camera_points, crs=pyproj.CRS.from_epsg("4326")
        )
        # Make sure the gdf has a gemetric CRS so there is no warping of the space
        camera_points = ensure_geometric_CRS(camera_points)
        # Extract the x, y points now in a geometric CRS
        camera_points_numpy = np.stack(
            camera_points.geometry.apply(lambda point: (point.x, point.y))
        )

        # Assign each camera to a cluster
        camera_cluster_IDs = KMeans(n_clusters=n_clusters).fit_predict(
            camera_points_numpy
        )
        if vis_clusters:
            plt.scatter(
                camera_points_numpy[:, 0],
                camera_points_numpy[:, 1],
                c=camera_cluster_IDs,
                cmap="tab20",
            )
            plt.show()

        # Do pix2face by cluster
        for cluster_ID in tqdm(range(n_clusters), desc="Chunks in mesh"):
            # Get indices of cameras for that cluster
            matching_camera_inds = np.where(cluster_ID == camera_cluster_IDs)[0]
            # Extract the rows in the dataframe for those IDs
            subset_camera_points = camera_points.iloc[matching_camera_inds]

            # TODO this could be accellerated by computing the membership for all points at the begining.
            # This would require computing all the ROIs (potentially-overlapping) for each region first. Then, finding all the non-overlapping
            # partition where each polygon corresponds to a set of ROIs. Then the membership for each vertex could be found for each polygon
            # and the membership in each ROI could be computed. This should be benchmarked though, because having more polygons than original
            # ROIs may actually lead to slower computations than doing it sequentially

            # Extract a sub mesh for a region around the camera points and also retain the indices into the original mesh
            sub_mesh_pv, _, face_IDs = self.select_mesh_ROI(
                region_of_interest=subset_camera_points,
                buffer_meters=buffer_dist_meters,
                return_original_IDs=True,
            )
            # Wrap this pyvista mesh in a photogrammetry mesh
            sub_mesh_TPM = TexturedPhotogrammetryMesh(sub_mesh_pv)

            # Get the segmentor camera set for the subset of the camera inds
            sub_camera_set = cameras.get_subset_cameras(matching_camera_inds)

            # Return the submesh as a Textured Photogrammetry Mesh, the subset of cameras, and the
            # face IDs mapping the faces in the sub mesh back to the full one
            yield sub_mesh_TPM, sub_camera_set, face_IDs

    def aggregate_projected_images(
        self,
        cameras: PhotogrammetryCamera | PhotogrammetryCameraSet,
        n_clusters: int = 8,
        buffer_dist_meters: float = 50,
        vis_clusters: bool = False,
        batch_size: int = 1,
        aggregate_img_scale: float = 1,
        **kwargs
    ):
        """Note that you cannot return_all

        Args:
            cameras (PhotogrammetryCamera | PhotogrammetryCameraSet): _description_
            batch_size (int, optional): _description_. Defaults to 1.
            aggregate_img_scale (float, optional): _description_. Defaults to 1.
            return_all (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        summed_projections = np.zeros(
            (self.pyvista_mesh.n_faces, cameras.n_image_channels()), dtype=float
        )
        projection_counts = np.zeros(self.pyvista_mesh.n_faces, dtype=int)

        # Create a generator to generate chunked meshes
        chunk_gen = self.get_mesh_chunks_for_cameras(
            cameras,
            n_clusters=n_clusters,
            buffer_dist_meters=buffer_dist_meters,
            vis_clusters=vis_clusters,
        )
        # Iterate over chunks in the mesh
        for sub_mesh_TPM, sub_camera_set, face_IDs in chunk_gen:
            # Aggregate the projections from a set of cameras corresponding to
            _, additional_information_submesh = sub_mesh_TPM.aggregate_projected_images(
                sub_camera_set,
                batch_size=batch_size,
                aggregate_img_scale=aggregate_img_scale,
                return_all=False,
                **kwargs
            )

            # Increment the summed predictions and counts
            # Make sure that nans don't propogate, since they should just be treated as zeros
            # TODO ensure this is correct
            summed_projections[face_IDs] = np.nansum(
                [
                    summed_projections[face_IDs],
                    additional_information_submesh["summed_projections"],
                ],
                axis=0,
            )
            projection_counts[face_IDs] = (
                projection_counts[face_IDs]
                + additional_information_submesh["projection_counts"]
            )

        # Same as the parent class
        no_projections = projection_counts == 0
        summed_projections[no_projections] = np.nan

        additional_information = {
            "projection_counts": projection_counts,
            "summed_projections": summed_projections,
        }

        average_projections = np.divide(
            summed_projections, np.expand_dims(projection_counts, 1)
        )

        return average_projections, additional_information
