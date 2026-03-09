"""Functions for computing neighborhood features."""

from typing import Literal, Optional, Union

import polars as pl
import spatial_image as si
from ngio.common import Roi
from ngio.images import Label
from ngio.images._masked_image import MaskedLabel

from abbott_features.features.constants import DensityParams
from abbott_features.features.neighborhood import aggregation_functions
from abbott_features.features.neighborhood.neighborhoods import NeighborhoodQueryObject
from abbott_features.fractal_tasks.fractal_utils import ensure_uint16, remap_label_ids

default_params = DensityParams()


def get_neighborhood_features(
    roi: Roi,
    label_image: Union[Label, MaskedLabel],
    label_img_mask: Optional[Union[Label, MaskedLabel]] = None,
    radius: tuple[float, ...] = default_params.radius,
    knn_distance: tuple[int, ...] = default_params.knn_distance,
    distance_to_closest_neighbor: bool = default_params.distance_to_closest_neighbor,
    delaunay: tuple[int, ...] = default_params.delaunay,
    touch: tuple[int, ...] = default_params.touch,
    distance_aggfuncs: tuple[int, ...] = default_params.distance_aggfuncs,
    adjacency_aggfuncs: tuple[int, ...] = default_params.adjacency_aggfuncs,
    index_columns: tuple[Literal["label", "label_image"], ...] = ("label",),
) -> pl.DataFrame:
    axes_names = label_image.axes
    pixel_sizes = label_image.pixel_size.as_dict()
    scale = label_image.pixel_size.zyx

    # Get the label image
    if isinstance(label_image, MaskedLabel):
        label_numpy = label_image.get_roi_masked_as_numpy(int(roi.name))
    else:
        label_numpy = label_image.get_roi_as_numpy(roi)

    # Relabel to uint16 if needed (itk requires values <= uint16 max)
    label_numpy, new_to_old = ensure_uint16(label_numpy)

    lbl = si.to_spatial_image(
        label_numpy,
        dims=axes_names,
        scale=pixel_sizes,
    )

    # Get the masking label image
    if label_img_mask is not None:
        if isinstance(label_img_mask, MaskedLabel):
            label_numpy_to = label_img_mask.get_roi_masked_as_numpy(int(roi.name))
        else:
            label_numpy_to = label_img_mask.get_roi_as_numpy(roi)

        # Relabel to uint16 if needed (itk requires values <= uint16 max)
        label_numpy_to, _ = ensure_uint16(label_numpy_to)

        mask = si.to_spatial_image(
            label_numpy_to,
            dims=axes_names,
            scale=pixel_sizes,
            name=label_img_mask.meta.name,
        )
    else:
        mask = None

    nq = NeighborhoodQueryObject.from_labelimage(lbl=lbl, mask=mask, scale=scale)
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

    # If relabeling was performed, restore original label IDs in the feature table.
    if new_to_old is not None:
        df = remap_label_ids(df, new_to_old)

    # add ROI column
    df = df.with_columns(pl.lit(roi.name).alias("ROI"))

    return df.select(pl.col(index_columns), pl.exclude(index_columns))
