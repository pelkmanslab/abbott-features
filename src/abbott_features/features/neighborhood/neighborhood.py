"""Functions for computing neighborhood features."""

from typing import Literal

import ngio
import polars as pl

from abbott_features.features.constants import DensityParams
from abbott_features.features.neighborhood import aggregation_functions
from abbott_features.features.neighborhood.neighborhoods import NeighborhoodQueryObject

default_params = DensityParams()


def get_neighborhood_features(
    label_image: ngio.images.Label,
    label_img_mask: ngio.images.Label,
    roi: ngio.common.Roi,
    radius: tuple[float, ...] = default_params.radius,
    knn_distance: tuple[int, ...] = default_params.knn_distance,
    distance_to_closest_neighbor: bool = default_params.distance_to_closest_neighbor,
    delaunay: tuple[int, ...] = default_params.delaunay,
    touch: tuple[int, ...] = default_params.touch,
    distance_aggfuncs: tuple[int, ...] = default_params.distance_aggfuncs,
    adjacency_aggfuncs: tuple[int, ...] = default_params.adjacency_aggfuncs,
    index_columns: tuple[Literal["label", "label_image"], ...] = ("label",),
) -> pl.DataFrame:
    nq = NeighborhoodQueryObject.from_labelimage(label_image, label_img_mask, roi)
    results = []
    distance_aggfuncs = [getattr(aggregation_functions, f) for f in distance_aggfuncs]
    adjacency_aggfuncs = [getattr(aggregation_functions, f) for f in adjacency_aggfuncs]

    # Compute object counts in radius
    results.append(
        nq.radius(radius, self_loops=False, distance=False).aggregate(
            adjacency_aggfuncs
        )
    )

    # Compute distance to closest neighbor
    if distance_to_closest_neighbor:
        results.append(
            nq.knn(k=1, self_loops=False, distance=True).aggregate_weights(
                aggregation_functions.Max
            )
        )

    # Compute distances to closest neighbors
    results.append(
        nq.knn(knn_distance, self_loops=False, distance=True).aggregate_weights(
            distance_aggfuncs
        )
    )

    # TODO: implement thresholding
    # Compute delaunay neighbor counts
    results.append(
        nq.delaunay(delaunay, self_loops=False).aggregate(adjacency_aggfuncs)
    )

    # TODO: implement thresholding
    # Compute touch neighbor counts
    results.append(nq.touch(touch, self_loops=False).aggregate(adjacency_aggfuncs))

    df = pl.concat(
        results,
        how="horizontal",
    )

    if "label" in index_columns:
        df = df.with_columns(nq.label.select(pl.col("label")))

    if "label_image" in index_columns:
        df = df.with_columns(pl.lit(label_image.meta.name).alias("label_image"))

    # add ROI column
    df = df.with_columns(pl.lit(roi.name).alias("ROI"))

    return df.select(pl.col(index_columns), pl.exclude(index_columns))
