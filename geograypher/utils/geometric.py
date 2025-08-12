import logging
import typing

import numpy as np
import pyvista as pv
import shapely
from shapely import simplify, unary_union
from tqdm import tqdm

logger = logging.getLogger("geometric")


def batched_unary_union(
    geometries: typing.List[shapely.Geometry],
    batch_size: int,
    grid_size: typing.Union[None, float] = None,
    subsequent_batch_size: int = 4,
    sort_by_loc: bool = False,
    simplify_tol: float = 0,
    verbose: bool = False,
) -> shapely.MultiPolygon:
    """Roughly replicate the functionality of shapely.unary_union using a batched implementation

    Args:
        geometries (typing.List[shapely.Geometry]): Geometries to aggregate
        batch_size (int): The batch size for the first aggregation
        grid_size (typing.Union[None, float]): grid size passed to unary_union
        subsequent_batch_size (int, optional): The batch size for subsequent (recursive) batches. Defaults to 4.
        sort_by_loc (bool, optional): Should the polygons be sorted by location to have a higher likelihood of merging. Defaults to False.
        simplify_tol (float, optional): How much to simplify in intermediate steps
        verbose (bool, optional): Should additional print outs be provided

    Returns:
        shapely.MultiPolygon: The merged multipolygon
    """

    # If the geoemtry is already one entry or empty
    if len(geometries) <= 1:
        # Run unary_union to ensure that edge cases such as empty geometries are handled in the expected way
        return unary_union(geometries)

    # Sort the polygons by their least x coordinates (any bound could be used)
    # The goal of this is to have a higher likelihood of grouping objects together and removing interior coordinates
    if sort_by_loc and batch_size < len(geometries):
        logger.error(f"Sorting the geometries with {len(geometries)} entries")
        geometries = sorted(geometries, key=lambda x: x.bounds[0])
        logger.error("Done sorting geometries")

    # TODO you could consider requesting a give number of points in the batch,
    # rather than number of objects.
    # TODO you could consider multiprocessing this since it's embarassingly parallel

    # Wrap the iteration in tqdm if requested, else just return it
    iteration_decorator = lambda x: (
        tqdm(x, desc=f"Computing batched unary union with batch size {batch_size}")
        if verbose
        else x
    )

    # Compute batched version
    batched_unions = []
    for i in iteration_decorator(
        range(0, len(geometries), batch_size),
    ):
        batch = geometries[i : i + batch_size]
        batched_unions.append(unary_union(batch, grid_size=grid_size))

    # Simplify the geometry to reduce the number of points
    # Don't do this if it would otherwise be returned as-is
    if simplify_tol > 0.0 and len(batched_unions) > 1:
        # Simplify then buffer, to make sure we don't have holes
        logger.info(
            f"Lengths before simplification {[len(bu.geoms) for bu in batched_unions]}"
        )
        batched_unions = [
            simplify(bu, simplify_tol).buffer(simplify_tol)
            for bu in tqdm(batched_unions, desc="simplifying polygons")
        ]
        logger.info(
            f"Lengths after simplification {[len(bu.geoms) for bu in batched_unions]}"
        )
    # Recurse this process until there's only one merged geometry
    # All future calls will use the subsequent_batch_size
    # TODO this batch size could be computed more inteligently, or sidestepped by requesting a number of points
    # Don't sort because this should already be sorted
    # Don't simplify because we don't want to repeatedly degrade the geometry
    return batched_unary_union(
        batched_unions,
        batch_size=subsequent_batch_size,
        grid_size=grid_size,
        subsequent_batch_size=subsequent_batch_size,
        sort_by_loc=False,
        simplify_tol=0.0,
    )


def get_scale_from_transform(transform: typing.Union[np.ndarray, None]):
    if transform is None:
        return 1

    if transform.shape != (4, 4):
        raise ValueError(f"Transform shape was {transform.shape}")

    transform_determinant = np.linalg.det(transform[:3, :3])
    scale_factor = np.cbrt(transform_determinant)
    return scale_factor


def unit_vector(vector):
    """Returns the unit vector of the vector."""
    return vector / np.linalg.norm(vector)


# https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python
def angle_between(v1, v2):
    """Returns the angle in radians between vectors 'v1' and 'v2'::

    >>> angle_between((1, 0, 0), (0, 1, 0))
    1.5707963267948966
    >>> angle_between((1, 0, 0), (1, 0, 0))
    0.0
    >>> angle_between((1, 0, 0), (-1, 0, 0))
    3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def orthogonal_projection(v1, v2):
    # component of v2 along v1
    scalar = np.dot(v1, v2) / np.linalg.norm(v1) ** 2
    return scalar * v1


def projection_onto_plane(v1, e1, e2):
    # Compute the projection of vector v1 onto the plane defined by e1, e2
    plane_normal = np.cross(e1, e2)
    out_of_plane_projection = orthogonal_projection(plane_normal, v1)
    in_plane_projection = v1 - out_of_plane_projection
    return in_plane_projection


def clip_line_segments(
    boundaries: typing.Tuple[pv.PolyData, pv.PolyData],
    origins: np.ndarray,
    directions: np.ndarray,
    image_indices: typing.Union[typing.List[int], np.ndarray],
    ray_limit: typing.Optional[float] = None,
) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Clips line segments between two boundary surfaces using ray tracing,
    keeping only the segments that intersect both surfaces.

    Args:
        boundaries (Tuple[pv.PolyData, pv.PolyData]):
            A tuple containing two PyVista PolyData surfaces representing the boundaries.
        origins (np.ndarray):
            Array of ray origin points (shape: [N, 3]).
        directions (np.ndarray):
            Array of ray direction vectors (shape: [N, 3]).
        image_indices (List[int]):
            List of indices associated with each ray, used for identification.
            Tracks which image each ray corresponded to.
        ray_limit (Optional[float], optional):
            If provided, segments longer than this value are dropped as invalid. Note
            that the filtered distance is from the **original ray start to the second
            boundary**. This is to mimic measuring from a camera (hypothetical ray
            source) to the ground (assuming the boundaries are given as [ceiling, floor]).

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            - (N, 3) array of start points of clipped segments. This will be the point
                that intersected boundaries[0]
            - (N, 3) array of end points of clipped segments. This will be the point
                that intersected boundaries[1]
            - (N, 3) array of direction vectors for the clipped segments.
            - (N,) array of image indices corresponding to the clipped segments.

    Raises:
        ValueError if various input requirements are not met.
    """

    # Input checking
    if len(boundaries) != 2:
        raise ValueError(f"2 boundaries required, not {len(boundaries)}")
    if any([not isinstance(b, pv.PolyData) for b in boundaries]):
        raise ValueError(f"pv.PolyData required, found {[type(b) for b in boundaries]}")
    if origins.shape != directions.shape:
        raise ValueError(
            f"origins and directions mismatched ({origins.shape} != {directions.shape})"
        )
    if origins.shape[1] != 3:
        raise ValueError(f"(N, 3) input arrays required, found {origins.shape}")
    if len(origins) != len(image_indices):
        raise ValueError(
            f"origins and image indices mismatched ({len(origins)} !="
            f" {len(image_indices)})"
        )

    # Handle the empty case
    if len(origins) == 0:
        return (
            origins.copy(),
            origins.copy(),
            directions.copy(),
            np.array(image_indices),
        )

    # Calculate ray/boundary intersections
    pts0, idx0, _ = boundaries[0].multi_ray_trace(
        origins=origins,
        directions=directions,
        first_point=True,
        retry=True,
    )
    pts1, idx1, _ = boundaries[1].multi_ray_trace(
        origins=origins,
        directions=directions,
        first_point=True,
        retry=True,
    )

    # Find the ray indices that were found to intersect with both surfaces
    matched_set = list(set(idx0).intersection(set(idx1)))

    # Iterate through the rays with intersections and update the starts and ends
    new_starts = []
    new_ends = []
    new_directions = []
    new_indices = []

    for ray_index in matched_set:
        pt0 = pts0[np.where(idx0 == ray_index)[0][0]]
        pt1 = pts1[np.where(idx1 == ray_index)[0][0]]

        # Limit the length of the origin → boundary[1] vector if requested
        if ray_limit is not None:
            vector = origins[ray_index] - pt1
            if np.linalg.norm(vector) > ray_limit:
                continue

        # Keep track of the outputs
        new_starts.append(pt0)
        new_ends.append(pt1)
        new_directions.append((pt1 - pt0) / np.linalg.norm(pt1 - pt0))
        new_indices.append(image_indices[ray_index])

    return (
        np.array(new_starts),
        np.array(new_ends),
        np.array(new_directions),
        np.array(new_indices),
    )
