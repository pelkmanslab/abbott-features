"""Pydantic models for advanced iterator configuration."""

from typing import Literal

from pydantic import BaseModel, Field


class MaskingConfiguration(BaseModel):
    """Masking configuration."""

    mode: Literal["Table Name", "Label Name"] = "Table Name"
    """
    Mode of masking to be applied. If "Table Name", the identifier refers to a
    masking table name.
    If "Table Name", the identifier refers to a masking table name.
    If "Label Name", the identifier refers to a label image name.
    """
    identifier: str | None = None
    """
    Name of the masking table or label image depending on the mode.
    """


class IteratorConfiguration(BaseModel):
    """Advanced Masking configuration."""

    masking: MaskingConfiguration | None = Field(
        default=None, title="Masking Iterator Configuration"
    )
    """
    If configured, the segmentation will be only saved within the mask region.
    will be applied based on the provided masking configuration.
    """
    roi_table: str | None = Field(default=None, title="Iterate Over ROIs")
    """
    Name of a ROI table. If provided, the segmentation will be performed for each ROI
    in the specified ROI table.
    """
