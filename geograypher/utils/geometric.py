import logging
import typing

import numpy as np
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
    simplify_tol=0,
    verbose: bool = False,
) -> shapely.MultiPolygon:
    """Roughly replicate the functionality of shapely.unary_union using a batched implementation

    Args:
        geometries (typing.List[shapely.Geometry]): Geometries to aggregate
        batch_size (int): The batch size for the first aggregation
        grid_size (typing.Union[None, float]): grid size passed to unary_union
        subsequent_batch_size (int, optional): The batch size for subsequent (recursive) batches. Defaults to 4.
        sort_by_loc (bool, optional): Should the polygons be sorted by location to have a higher likelihood of merging. Defaults to False.
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
