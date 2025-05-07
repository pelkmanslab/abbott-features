"""Functions to extract intensity features."""

from typing import TypeAlias

import polars as pl

from abbott_features.features._base import get_si_features_df
from abbott_features.features.constants import IntensityFeature
from abbott_features.features.types import LabelImage, SpatialImage

IntensityFeaturesLike: TypeAlias = tuple[IntensityFeature, ...] | tuple[str, ...]


def get_intensity_features(
    label_image: LabelImage,
    intensity_image: SpatialImage,
    ROI_id: str,
    features: IntensityFeaturesLike = tuple(IntensityFeature),
) -> pl.DataFrame:
    """Get intensity features for a given label image and intensity image."""
    valid_features = tuple(str(IntensityFeature(e)) for e in features)
    feature_table = get_si_features_df(
        label_image, intensity_image, props=valid_features, named_features=True
    )
    # add ROI column
    if ROI_id is not None:
        feature_table = feature_table.with_columns(pl.lit(ROI_id).alias("ROI"))
    return feature_table
