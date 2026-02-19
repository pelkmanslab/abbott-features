"""Function to extract features from label image."""

from typing import TypeAlias, Union

import numpy as np
import polars as pl
import spatial_image as si
from ngio.common import Roi
from ngio.images import Label
from ngio.images._masked_image import MaskedLabel

from abbott_features.features._base import get_si_features_df
from abbott_features.features.constants import (
    DefaultLabelFeature,
    LabelFeature,
    PositionAndOrientationLabelFeature,
)
from abbott_features.features.types import LabelImage

LabelFeatureLike: TypeAlias = tuple[LabelFeature, ...] | tuple[str, ...]


def get_label_features(
    label_image: Union[Label, MaskedLabel],
    roi: Roi,
    features: LabelFeatureLike = tuple(DefaultLabelFeature),
) -> pl.DataFrame:
    """Get label features from a label image."""
    axes_names = label_image.axes
    pixel_sizes = label_image.pixel_size.as_dict()

    if isinstance(label_image, MaskedLabel):
        label_numpy = label_image.get_roi_masked_as_numpy(int(roi.name)).astype(
            np.uint16
        )
    else:
        label_numpy = label_image.get_roi_as_numpy(roi).astype(np.uint16)

    label_spatial_image = si.to_spatial_image(
        label_numpy,
        dims=axes_names,
        scale=pixel_sizes,
    )

    valid_label_features = tuple(str(LabelFeature(e)) for e in features)
    feature_table = get_si_features_df(
        label_spatial_image, props=valid_label_features, named_features=True
    )
    # add ROI column
    feature_table = feature_table.with_columns(pl.lit(roi.name).alias("ROI"))
    return feature_table


def get_centroids(label_image: LabelImage) -> pl.DataFrame:
    """Get centroids of label objects."""
    return get_si_features_df(label_image, props=("Centroid",), named_features=True)


def get_position_and_orientation_features(label_image: LabelImage) -> pl.DataFrame:
    """Get position and orientation features of label objects."""
    return get_si_features_df(
        label_image,
        props=tuple(PositionAndOrientationLabelFeature),
        named_features=True,
    )
