import logging
import typing

import shapely
from shapely import unary_union
from tqdm import tqdm

logger = logging.getLogger("geometric")


def batched_unary_union(
    geometries: typing.List[shapely.Geometry],
    batch_size: int,
    subsequent_batch_size: int = 4,
    sort_by_loc: bool = False,
) -> shapely.MultiPolygon:
    """Roughly replicate the functionality of shapely.unary_union using a batched implementation

    Args:
        geometries (typing.List[shapely.Geometry]): Geometries to aggregate
        batch_size (int): The batch size for the first aggregation
        subsequent_batch_size (int, optional): The batch size for subsequent (recursive) batches. Defaults to 4.
        sort_by_loc (bool, optional): Should the polygons be sorted by location to have a higher likelihood of merging. Defaults to False.

    Returns:
        shapely.MultiPolygon: The merged multipolygon
    """

    # If the geoemtry is already one entry or empty
    if len(geometries) <= 1:
        # Run unary_union to ensure that edge cases such as empty geometries are handled in the expected way
        return unary_union(geometries)

    # Sort the polygons by their least x coordinates (any bound could be used)
    # The goal of this is to have a higher likelihood of grouping objects together and removing interior coordinates
    if sort_by_loc:
        logger.error("Sorting the geometries")
        geometries = sorted(geometries, key=lambda x: x.bounds[0])
        logger.error("Done sorting geometries")

    # TODO you could consider requesting a give number of points in the batch,
    # rather than number of objects.
    # TODO you could consider multiprocessing this since it's embarassingly parallel

    # Compute batched version
    batched_unions = []
    for i in tqdm(
        range(0, len(geometries), batch_size),
        desc=f"Computing batched unary union with batch size {batch_size}",
    ):
        batch = geometries[i : i + batch_size]
        batched_unions.append(unary_union(batch))

    # TODO consider simplifying the geometry with shapely.simpify

    # Recurse this process until there's only one merged geometry
    # All future calls will use the subsequent_batch_size
    # TODO this batch size could be computed more inteligently, or sidestepped by requesting a number of points
    # Don't sort because this should already be roughly sorted
    return batched_unary_union(
        batched_unions,
        batch_size=subsequent_batch_size,
        subsequent_batch_size=subsequent_batch_size,
        sort_by_loc=False,
    )
