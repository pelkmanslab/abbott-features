"""Functions to extract intensity features."""

from typing import TypeAlias, Union

import ngio
import polars as pl
import spatial_image as si

from abbott_features.features._base import get_si_features_df
from abbott_features.features.constants import IntensityFeature

IntensityFeaturesLike: TypeAlias = tuple[IntensityFeature, ...] | tuple[str, ...]


def get_intensity_features(
    label_image: Union[ngio.images.label.Label, ngio.images.masked_image.MaskedLabel],
    images: ngio.images.image.Image,
    channel_label: str,
    roi: ngio.common._roi.Roi,
    features: IntensityFeaturesLike = tuple(IntensityFeature),
) -> pl.DataFrame:
    """Get intensity features for a given label image and intensity image."""
    axes_names = label_image.axes_mapper.on_disk_axes_names
    pixel_sizes = label_image.pixel_size.as_dict()

    if isinstance(label_image, ngio.images.masked_image.MaskedLabel):
        label_numpy = label_image.get_roi_masked(int(roi.name)).astype("uint16")
    else:
        label_numpy = label_image.get_roi(roi).astype("uint16")

    label_spatial_image = si.to_spatial_image(
        label_numpy,
        dims=axes_names,
        scale=pixel_sizes,
    )

    channel_idx = images.meta.get_channel_idx(label=channel_label)
    image_numpy = (
        images.get_roi(roi, c=channel_idx, mode="numpy").astype("uint16").squeeze()
    )
    intensity_spatial_image = si.to_spatial_image(
        image_numpy,
        dims=axes_names,
        scale=pixel_sizes,
        name=channel_label,
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
