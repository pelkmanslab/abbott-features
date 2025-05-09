# Copyright 2022 (C) Friedrich Miescher Institute for Biomedical Research and
# University of Zurich
#
# Original authors:
# Tommaso Comparin <tommaso.comparin@exact-lab.it>
#
# This file is part of Fractal and was originally developed by eXact lab S.r.l.
# <exact-lab.it> under contract with Liberali Lab from the Friedrich Miescher
# Institute for Biomedical Research and Pelkmans Lab from the University of
# Zurich.
"""Helper functions to address channels via OME-NGFF/OMERO metadata."""

from typing import Self

from pydantic import BaseModel, model_validator


class ChannelInputModel(BaseModel):
    """A channel which is specified by either `wavelength_id` or `label`.

    This model is similar to `OmeroChannel`, but it is used for
    task-function arguments (and for generating appropriate JSON schemas).

    Attributes:
        wavelength_id: Unique ID for the channel wavelength, e.g. `A01_C01`.
            Can only be specified if label is not set.
        label: Name of the channel. Can only be specified if wavelength_id is
            not set.
    """

    wavelength_id: str | None = None
    label: str | None = None

    @model_validator(mode="after")
    def mutually_exclusive_channel_attributes(self: Self) -> Self:
        """Check that either `label` or `wavelength_id` is set."""
        wavelength_id = self.wavelength_id
        label = self.label

        if wavelength_id and label:
            raise ValueError(
                "`wavelength_id` and `label` cannot be both set "
                f"(given {wavelength_id=} and {label=})."
            )
        if wavelength_id is None and label is None:
            raise ValueError("`wavelength_id` and `label` cannot be both `None`")
        return self


class ChannelPairInputModel(BaseModel):
    """Get channel pair to measure colocalization features.

    Attributes:
        channel0: First channel to measure colocalization features.
        channel1: Second channel to measure colocalization features.
    """

    channel0: ChannelInputModel
    channel1: ChannelInputModel


class IntensityFeaturesInputModel(BaseModel):
    """Get intensity features to measure.

    Attributes:
        channels_to_include: Channels to include in the measurement.
        channels_to_exclude: Channels to exclude from the measurement.
    """

    channels_to_include: list[ChannelInputModel] = None
    channels_to_exclude: list[ChannelInputModel] = None

    @model_validator(mode="after")
    def mutually_exclusive_channel_attributes(self: Self) -> Self:
        """Check that either `channels_to_include` or `channels_to_exclude` is set."""
        channels_to_include = self.channels_to_include
        channels_to_exclude = self.channels_to_exclude

        if channels_to_include and channels_to_exclude:
            raise ValueError(
                "`channels_to_include` and `channels_to_exclude` cannot be both set "
                f"(given {channels_to_include=} and {channels_to_exclude=})."
            )
        return self


class DistanceFeaturesInputModel(BaseModel):
    """Get label_name of label image to measure distance to.

    Attributes:
        label_name_to: Name of the label image to measure distance
            to e.g. "embryo" or "organoid".
    """

    label_name_to: str


class ColocalizationFeaturesInputModel(BaseModel):
    """Get channel pair(s) to measure colocalization features.

    Attributes:
        channel_pair: Name of the channel pair to measure colocalization features.
    """

    channel_pair: list[ChannelPairInputModel]


class NeighborhoodFeaturesInputModel(BaseModel):
    """Get label_name of label image to measure neighborhood in.

    Attributes:
        measure: Whether to measure neighborhood features or not.
        label_img_mask: Name of the label image to measure neighborhood
            in e.g. "embryo" or "organoid".
    """

    measure: bool = False
    label_img_mask: str
