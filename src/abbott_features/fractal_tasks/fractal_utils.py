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
