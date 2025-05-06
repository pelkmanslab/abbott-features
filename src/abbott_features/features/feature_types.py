"""Feature types for the Abbott features module."""

from dataclasses import field
from typing import (
    NewType,
    Self,
    TypeAlias,
)

from pydantic.dataclasses import dataclass as pdataclass

RoiID = NewType("RoiID", str)
ImageID = NewType("ImageID", str)
ChannelID = NewType("ChannelID", str)
LabelImageID = NewType("LabelImageID", str)
LabelObjectID = NewType("LabelObjectID", tuple[LabelImageID, int])
PointsID = NewType("PointsID", str)
ShapesID = NewType("ShapesID", str)
TableID = NewType("TableID", str)
FeatureID = NewType("FeatureID", tuple[TableID, str])

RegionID: TypeAlias = LabelImageID | PointsID | ShapesID
ResourceID: TypeAlias = ChannelID | LabelImageID | LabelObjectID | TableID | FeatureID

ElementID: TypeAlias = (
    RoiID
    | ChannelID
    | LabelImageID
    | LabelObjectID
    | PointsID
    | ShapesID
    | TableID
    | FeatureID
    | RegionID
)


@pdataclass
class Resources:
    """Resources required for feature measurement."""

    label_images: set[LabelImageID] = field(default_factory=set)
    channels: set[ChannelID] = field(default_factory=set)
    channel_pairs: set[tuple[ChannelID, ChannelID]] = field(default_factory=set)
    label_objects: set[LabelObjectID] = field(default_factory=set)
    tables: set[TableID] = field(default_factory=set)
    features: set[FeatureID] = field(default_factory=set)

    def union(self, *others: "Resources") -> "Resources":
        """Combine resources from multiple sources."""
        return Resources(
            label_images=self.label_images.union(
                *[other.label_images for other in others]
            ),
            channels=self.channels.union(*[other.channels for other in others]),
            channel_pairs=self.channels.union(
                *[other.channel_pairs for other in others]
            ),
            tables=self.tables.union(*[other.tables for other in others]),
            features=self.features.union(*[other.features for other in others]),
        )

    def validate_available(self, other: "Resources") -> Self:
        """Validate that the resources in `other` are available in this instance."""
        _validate_resource_query(self, other)
        return self

    def __lt__(self, other):
        """Compare resources based on their contents."""
        keys = Resources.__dataclass_fields__.keys()
        return tuple(sorted(getattr(self, key)) for key in keys) < tuple(
            sorted(getattr(other, key)) for key in keys
        )


def _validate_resource_query(resources: Resources, available_resources: Resources):
    for k in Resources.__dataclass_fields__.keys():
        missing_resources = getattr(resources, k).difference(
            getattr(available_resources, k)
        )
        assert (
            missing_resources == set()
        ), f"Missing resources: `{missing_resources}` not found in {k}"
