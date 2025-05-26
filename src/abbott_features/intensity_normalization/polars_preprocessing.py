"""Polars preprocessing functions for intensity normalization."""

import logging
from functools import reduce
from operator import or_

import polars as pl
import polars.selectors as cs
from polars.type_aliases import SelectorType

from abbott_features.intensity_normalization.polars_utils import AnyFrameT, join

logger = logging.getLogger(__name__)

META_COLUMNS = reduce(
    or_,
    [
        cs.matches(f"^{feature}$")
        for feature in [
            "roi",
            "parent.embryoRaw",
            "nucleiRaw3_Count",
            "log2_nucleiRaw3_Count",
            "cycle",
            "well",
            "is_control_well",
            "control_well_for_acquisition",
            "embryo",
        ]
    ],
)


def hierarchy_aggregate_count(
    df: pl.DataFrame,
    parent_selector=cs.by_name("well") | cs.by_name("ROI"),
    object_column=cs.by_name("object"),
) -> pl.DataFrame:
    parent_columns = cs.expand_selector(df, parent_selector)
    objects_to_aggregate = df.select(object_column).unique().to_series()
    res = []
    for object_ in objects_to_aggregate:
        res.append(
            df.select(parent_selector, object_column)
            .filter(object_column == object_)
            .group_by(parent_selector)
            .agg(pl.col("object").count().alias(f"{object_}_Count"))
        )
    return join(res, on=parent_columns).sort(by=parent_columns)


def get_metadata(
    df: AnyFrameT,
    objects_to_count: tuple[str],
    parent_selector: SelectorType = cs.by_name("well") | cs.by_name("ROI"),
    control_wells: list[str] | None = None,
    object_column: SelectorType = cs.by_name("object"),
) -> pl.DataFrame:
    df_sub = df.filter(object_column.is_in(objects_to_count))
    df_meta = (
        hierarchy_aggregate_count(df_sub, parent_selector, object_column=object_column)
        .with_columns(
            cs.matches("_Count").log(base=2).cast(pl.Float32).name.prefix("log2_")
        )
        .with_columns(
            pl.col("well").str.slice(0, 1).alias("row"),
            pl.col("well").str.slice(1).alias("col"),
        )
    )
    if control_wells is not None:
        df_meta = df_meta.with_columns(
            [
                pl.col("well").is_in(control_wells).alias("is_control_well"),
            ]
        )
    return df_meta
