"""Functions to select features based on pattern etc. using Polars selectors."""

from functools import reduce
from itertools import product
from operator import or_
from typing import TYPE_CHECKING

import polars.selectors as cs
from attrs import define, field, make_class

from abbott_features.features.constants import (
    ColocalizationFeature,
    DistanceFeature,
    DistanceFunction,
    IntensityFeature,
    LabelFeature,
)

if TYPE_CHECKING:
    from polars.type_aliases import SelectorType


LABEL_PATTERN = r"^{}((.lower)|(.upper)|(Direction)|(Vertices)|(Size)|(Origin))?(\W[abc])?(\W[xyz])?$"  # noqa: E501

CHANNEL_PATTERN = r"([a-zA-Z0-9-]+)[_](\d+)"
INTENSITY_PATTERN = rf"^({CHANNEL_PATTERN}_)?{'{}'}(\W[abc])?(\W[xyz])?$"

CHANNEL_PAIR_PATTERN = rf"{CHANNEL_PATTERN}\W{CHANNEL_PATTERN}"

COLOC_PATTERN = f"^({CHANNEL_PAIR_PATTERN}_)?{'{}'}$"

DISTANCE_PATTERN = r"^([a-zA-Z0-9_]+_)?{}{}(\W[xyz])?$"

LABEL = reduce(or_, [cs.matches(LABEL_PATTERN.format(f)) for f in LabelFeature])
INTENSITY = reduce(
    or_, [cs.matches(INTENSITY_PATTERN.format(f)) for f in IntensityFeature]
)
COLOC = reduce(
    or_, [cs.matches(COLOC_PATTERN.format(f)) for f in ColocalizationFeature]
)
DISTANCE = reduce(
    or_,
    [
        cs.matches(DISTANCE_PATTERN.format(f, t))
        for f, t in product(DistanceFeature, DistanceFunction)
    ],
)
DENSITY = (
    cs.matches(r"^DELAUNAY:\d+_Count$")
    | cs.matches(r"^TOUCH:\d+_Count$")
    | cs.matches(r"^RAD:\d+(\W\d+)?_Count$")
    | cs.matches(r"^KNNd:\d+_")
)
NEIGHBORHOOD = (
    cs.matches(r"^DELAUNAY(s)?:\d+_.*$")
    | cs.matches(r"^TOUCH(s)?:\d+_.*$")
    | cs.matches(r"^RAD(s)?:\d+(\W\d+)?_.*$")
    | cs.matches(r"^KNN(d)?(s)?:\d+_.*$")
)
FEATURES = LABEL | INTENSITY | COLOC | DISTANCE | DENSITY | NEIGHBORHOOD

HIERARCHY = cs.matches(r"^parent\W.+$")

FULL_INDEX = (
    cs.matches("^well$")
    | cs.matches("^ROI$")
    | cs.matches("^index$")
    | HIERARCHY
    | cs.matches("^object$")
    | cs.matches("^label$")
    | cs.matches("^channel$")
    | cs.matches("^stain$")
    | cs.matches("^acquisition$")
    | cs.matches("^path_in_well$")
    | cs.matches("^channel_pair$")
)

SITE_INDEX = (
    cs.matches("^ROI$") | HIERARCHY | cs.matches("^object$") | cs.matches("^label$")
)

OBJECT_INDEX = SITE_INDEX - HIERARCHY


@define(frozen=True)
class MySelector:
    empty: "SelectorType" = ~cs.all()
    index: "SelectorType" = FULL_INDEX
    site_index: "SelectorType" = SITE_INDEX
    obj_index: "SelectorType" = OBJECT_INDEX
    meta: "SelectorType" = ~FEATURES
    hierarchy: "SelectorType" = HIERARCHY
    label: "SelectorType" = LABEL
    intensity: "SelectorType" = INTENSITY
    correlation: "SelectorType" = COLOC
    distance: "SelectorType" = DISTANCE
    density: "SelectorType" = DENSITY
    neighborhood: "SelectorType" = NEIGHBORHOOD
    features: "SelectorType" = FEATURES

    def expand(self, df):
        return expand_selector(df, self)


ExpandedSelector = make_class(
    "ExpandedSelector",
    attrs={
        attr.name: field(type=tuple[str, ...]) for attr in MySelector.__attrs_attrs__
    },
)


def expand_selector(df, sel: MySelector) -> ExpandedSelector:
    return ExpandedSelector(
        **{
            attr.name: cs.expand_selector(df, getattr(sel, attr.name))
            for attr in sel.__attrs_attrs__
        }
    )


sel = MySelector()
