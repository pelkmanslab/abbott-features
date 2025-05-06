"""Types of SpatialImage."""

from typing import TypeAlias

from spatial_image import SpatialImage

LabelImage: TypeAlias = SpatialImage
MultichannelLabelImage: TypeAlias = SpatialImage
BinaryImage: TypeAlias = SpatialImage
DistanceTransform: TypeAlias = SpatialImage
