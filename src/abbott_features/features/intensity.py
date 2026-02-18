"""Functions to extract intensity features."""

from typing import TypeAlias, Union

import numpy as np
import polars as pl
import spatial_image as si
from ngio.common import Roi
from ngio.images import Image, Label
from ngio.images._masked_image import MaskedImage, MaskedLabel

from abbott_features.features._base import get_si_features_df
from abbott_features.features.constants import IntensityFeature
from abbott_features.intensity_normalization.models import (
    apply_t_decay_factor,
    apply_z_decay_models,
)

IntensityFeaturesLike: TypeAlias = tuple[IntensityFeature, ...] | tuple[str, ...]


def get_intensity_features(
    label_image: Union[Label, MaskedLabel],
    images: Union[Image, MaskedImage],
    channel_label: str,
    roi: Roi,
    kwargs_decay_corr: dict,
    features: IntensityFeaturesLike = tuple(IntensityFeature),
) -> pl.DataFrame:
    """Get intensity features for a given label image and intensity image."""
    axes_names = label_image.axes
    pixel_sizes = label_image.pixel_size.as_dict()

    channel_idx = images.get_channel_idx(channel_label=channel_label)

    if isinstance(label_image, MaskedLabel):
        label_numpy = label_image.get_roi_masked(int(roi.name)).astype(np.uint16)
    else:
        label_numpy = label_image.get_roi(roi).astype(np.uint16)

    image_numpy = (images.get_roi(roi, c=channel_idx)).astype(np.uint16).squeeze()

    label_spatial_image = si.to_spatial_image(
        label_numpy,
        dims=axes_names,
        scale=pixel_sizes,
    )
    label_spatial_image.attrs["scale_dict"] = pixel_sizes

    intensity_spatial_image = si.to_spatial_image(
        image_numpy,
        dims=axes_names,
        scale=pixel_sizes,
        name=channel_label,
    )

    # Apply corrections if provided
    if kwargs_decay_corr["z_decay_correction"] is not None:
        z_decay_model = kwargs_decay_corr["z_decay_correction"]
        intensity_spatial_image = apply_z_decay_models(
            z_decay_model,
            intensity_spatial_image,
            label_spatial_image,
        )

    if kwargs_decay_corr["t_decay_correction_df"] is not None:
        correction_factors_df = kwargs_decay_corr["t_decay_correction_df"]
        intensity_spatial_image = apply_t_decay_factor(
            intensity_spatial_image, correction_factors_df, ROI_id=roi.name
        )

    valid_features = tuple(str(IntensityFeature(e)) for e in features)
    feature_table = get_si_features_df(
        label_spatial_image,
        intensity_spatial_image,
        props=valid_features,
        named_features=True,
    )
    # add ROI column
    feature_table = feature_table.with_columns(pl.lit(roi.name).alias("ROI"))
    return feature_table
