"""Feature queries for label images."""

from typing import TypeAlias

import polars as pl

from abbott_features.features._base import get_si_features_df
from abbott_features.features.constants import (
    DefaultLabelFeature,
    LabelFeature,
    PositionAndOrientationLabelFeature,
)
from abbott_features.features.types import LabelImage

LabelFeatureLike: TypeAlias = tuple[LabelFeature, ...] | tuple[str, ...]


def get_label_features(
    label_image: LabelImage,
    ROI_id: int | None = None,
    features: LabelFeatureLike = tuple(DefaultLabelFeature),
) -> "pl.DataFrame":
    """Get label features from a label image."""
    valid_label_features = tuple(str(LabelFeature(e)) for e in features)
    feature_table = get_si_features_df(
        label_image, props=valid_label_features, named_features=True
    )
    # add ROI column
    if ROI_id is not None:
        feature_table = feature_table.with_columns(pl.lit(ROI_id).alias("ROI"))
    return feature_table


def get_centroids(label_image: LabelImage) -> "pl.DataFrame":
    """Get centroids of label objects."""
    return get_si_features_df(label_image, props=("Centroid",), named_features=True)


def get_position_and_orientation_features(label_image: LabelImage) -> "pl.DataFrame":
    """Get position and orientation features of label objects."""
    return get_si_features_df(
        label_image,
        props=tuple(PositionAndOrientationLabelFeature),
        named_features=True,
    )
