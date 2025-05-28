"""Utils functions for polars DataFrames."""

import logging
import math
import os
import re
from collections import defaultdict
from collections.abc import Iterable, Sequence
from functools import reduce
from operator import add
from os import PathLike
from pathlib import Path
from typing import Any, Callable, Literal, TypeAlias, TypeGuard, TypeVar

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import polars.selectors as cs
import seaborn as sns
from polars.type_aliases import JoinStrategy, SelectorType
from strenum import StrEnum
from toolz.dicttoolz import valmap

from abbott_features.intensity_normalization.models import (
    Exp,
    ExpNoOffset,
    Linear,
    LogLinear,
)
from abbott_features.intensity_normalization.polars_selector import sel

logger = logging.Logger(__name__)

COLUMN_CASTS = {
    cs.matches("^acquisition$"): pl.UInt8,
    cs.matches("Count$|^label$|^parent"): pl.UInt32,
    cs.matches("^BoundingBox$"): pl.Struct(
        {
            **{f"lower-{e}": pl.Int32 for e in ["x", "y", "z"]},
            **{f"upper-{e}": pl.Int32 for e in ["x", "y", "z"]},
        }
    ),
    cs.matches("Index$"): pl.Struct({e: pl.Int32 for e in ["x", "y", "z"]}),
}
FULL_INDEX_COLUMNS = ["ROI", "object", "label", "channel", "stain", "acqusition"]
CAT_DTYPE_COLUMNS = cs.matches("^ROI$|^object$|^channel$|^stain$")
STRUCT_INT32_INT32_INT32_COLUMNS = cs.by_name("BoundingBox")
STRUCT_INT32_INT32_COLUMNS = cs.matches("Index$")


INDEX = ["ROI", "object", "label"]

OBJECT_INDEX_NAMES = ["ROI", "object", "label"]
OBJECT_INDEX_COLUMNS = [pl.col(p) for p in OBJECT_INDEX_NAMES]
NOT_OBJECT_INDEX_COLUMNS = [pl.exclude(OBJECT_INDEX_NAMES)]

CORR_FEATURE_NAMES = {
    "PearsonR",
    "SpearmanR",
    "KendallTau",
}

CORR_FEATURE_PATTERNS = {f"^.*{e}$" for e in CORR_FEATURE_NAMES}

INTENSITY_FEATURE_NAMES = {
    "CenterOfGravity",
    "Kurtosis",
    "Maximum",
    "MaximumIndex",
    "Mean",
    "Median",
    "Minimum",
    "MinimumIndex",
    "Skewness",
    "StandardDeviation",
    "Sum",
    "Variance",
    "WeightedElongation",
    "WeightedFlatness",
    "WeightedPrincipalAxes",
    "WeightedPrincipalMoments",
}

INTENSITY_FEATURE_PATTERNS = {
    f"^.*{e}(-lower|-upper)?(-[a-c])?(-[x-z])?$" for e in INTENSITY_FEATURE_NAMES
}
INTENSITY_FEATURE_COLUMNS = [pl.col(e) for e in INTENSITY_FEATURE_PATTERNS]

BOUNDING_BOX_STRUCT_COLUMN = "BoundingBox"
BOUNDING_BOX_COLUMNS = [
    "lower-x",
    "upper-x",
    "lower-y",
    "upper-y",
    "lower-z",
    "upper-z",
]

DEBUG = True

AnyFrameT = TypeVar("AnyFrameT", pl.DataFrame, pl.LazyFrame)
FrameOrLazy: TypeAlias = pl.DataFrame | pl.LazyFrame


def log(
    df: pl.DataFrame,
    message: str | None = "shape:",
    func: Callable[[pl.DataFrame], str] = lambda x: f"{x.shape}",
    **kwargs,
):
    introspection = func(df, **kwargs)
    out_message = f"{message} {introspection}"
    logger.info(out_message)
    return df


def id_(df: pl.DataFrame, **kwargs) -> pl.DataFrame:
    return df


def show(df: pl.DataFrame) -> pl.DataFrame:
    print(df)
    print()
    return df


def debug(df: pl.DataFrame, message: str = "", debug=DEBUG) -> pl.DataFrame:
    if debug:
        if message:
            print(f">>> {message}")
        print(df)
        print()
    return df


def lower_upper_iqr(series, q_lower=0.25, q_upper=0.75):
    ql = series.quantile(q_lower)
    qu = series.quantile(q_upper)
    iqr = qu - ql
    return ql, qu, iqr


def iqr(series: pl.Series, ql=0.25, qu=0.75, r=0.0):
    ql, qu, iqr = lower_upper_iqr(series, q_lower=ql, q_upper=qu)
    return (ql - r * iqr, qu + r * iqr)


def drop_structs(df: pl.DataFrame) -> pl.DataFrame:
    return df.select(
        [
            column
            for column, dtype in zip(df.columns, df.dtypes)
            if not isinstance(dtype, pl.Struct)
        ]
    )


def get_well_name(full_path: Path, ome_zarr_fld: Path) -> str:
    """Extracts the well name (e.g., 'B03') from a full OME-Zarr path.

    Parameters:
        full_path (Path): Full path to a file or directory under the OME-Zarr folder.
        ome_zarr_fld (Path): Base path to the OME-Zarr folder e.g.
            /path/to_ome_zarr_fld/

    Returns:
        str: Well name in the format '<row><column>', e.g., 'B03'.
    """
    relative_parts = full_path.relative_to(ome_zarr_fld).parts
    row = relative_parts[0]
    col = relative_parts[1]
    return f"{row}{col}"


def to_tall(
    df: pl.DataFrame,
    index_key: str,
):
    """Convert a polars DataFrame to a tall format."""
    # Get the columns to aggregate (excluding index_key)
    cols_to_agg = [col for col in df.columns if col != index_key]

    # Build expressions for each column: drop nulls and take the first value
    agg_exprs = [pl.col(c).drop_nulls().first().alias(c) for c in cols_to_agg]

    # Perform the groupby and aggregation
    result = df.group_by(index_key).agg(agg_exprs)
    result = result.with_columns(pl.col("label").cast(pl.Int32)).sort(by=index_key)
    return result


# NOT PERFORMANT WITH SMALL FRAGMENTED TABLES!
def scan_table(root: PathLike[str], _object: str = "") -> pl.LazyFrame:
    fns = list(Path(root).rglob(f"{_object}*.parquet"))
    tables = [
        pl.scan_parquet(fn).with_columns(
            pl.lit(fn.parent.name).alias("roi"), pl.lit(fn.stem).alias("object")
        )
        for fn in fns
    ]

    return pl.concat(tables, how="diagonal")


def column_null_count(
    df: pl.DataFrame, drop_zero_columns=False, sort=True
) -> pl.DataFrame:
    df_null_counts = (
        df.with_columns(cs.by_dtype(pl.Float32, pl.Float64))
        .fill_nan(None)
        .describe()
        .filter(pl.col("statistic") == "null_count")
        .select(pl.exclude("statistic").cast(pl.Int64))
        .transpose(
            include_header=True, header_name="feature", column_names=["null_count"]
        )
        .with_columns((pl.col("null_count") / df.height).alias("null_percentage"))
    )
    if sort:
        df_null_counts = df_null_counts.sort("null_count", descending=True)
    if drop_zero_columns:
        return df_null_counts.filter(pl.col("null_count") > 0)
    return df_null_counts


def drop_null_columns(
    df: pl.DataFrame,
    columns: SelectorType = cs.all(),
    strategy: Literal["any", "all", "perc"] | float = "all",
    allowed_percentage=0.3,
    include_nan=True,
) -> pl.DataFrame:
    if include_nan:
        df_nan = df.with_columns(cs.by_dtype(pl.Float32, pl.Float64).fill_nan(None))
    else:
        df_nan = df
    if strategy == "all":
        valid_col = df_nan.select(columns.is_null().all().not_())
    elif strategy == "any":
        valid_col = df_nan.select(columns.is_null().any().not_())
    elif strategy == "perc":
        null_percentage = df_nan.select(columns.is_null().sum() / columns.count().sum())
        valid_col = null_percentage <= allowed_percentage
    keep_columns = (
        valid_col.transpose(
            include_header=True, header_name="feature", column_names=["keep"]
        )
        .filter(pl.col("keep"))["feature"]
        .to_list()
    )
    passthrough_columns = df_nan.select(~columns).columns
    return df.select(passthrough_columns + keep_columns)


def drop_null_rows(
    df: pl.DataFrame,
    columns: SelectorType = cs.all(),
    strategy: Literal["any", "all", "perc"] | float = "all",
    allowed_percentage=0.3,
    include_nan=True,
) -> pl.DataFrame:
    if include_nan:
        df_nan = df.with_columns(cs.by_dtype(pl.Float32, pl.Float64).fill_nan(None))
    else:
        df_nan = df
    if strategy == "all":
        row_filter_pred = ~pl.all_horizontal(
            pl.col(cs.expand_selector(df_nan, columns)).is_null()
        )
    elif strategy == "any":
        row_filter_pred = ~pl.any_horizontal(columns.is_null())
    elif strategy == "perc":
        null_percentage = pl.sum_horizontal(columns.is_null()) / len(
            cs.expand_selector(df_nan, columns)
        )
        row_filter_pred = null_percentage <= allowed_percentage
    return df_nan.filter(row_filter_pred)


def _unnest_renaming_expression(df: pl.DataFrame, col: str, sep: str = "-") -> pl.Expr:
    return pl.col(col).struct.rename_fields(
        [f"{col}{sep}{f.name}" for f in df.schema.get(col).fields]
    )


def unnest_structs(
    df: pl.DataFrame, cols: str | Sequence[str], sep: str = "-"
) -> pl.DataFrame:
    if isinstance(cols, str):
        cols = (cols,)
    return df.select(
        [
            pl.exclude(cols),
            *[_unnest_renaming_expression(df, col, sep=sep) for col in cols],
        ]
    ).unnest(cols)


def unnest_all_structs(df: pl.DataFrame, sep: str = ".") -> pl.DataFrame:
    cols = [col for col, dtype in zip(df.columns, df.dtypes) if dtype == pl.Struct]
    return unnest_structs(df, cols, sep=sep)


POLARS_CONFIG = {"large": {"fmt_str_lengths": 50, "tbl_rows": 30}}


class NestingStrategy(StrEnum):
    SPLIT = "split"
    IGNORE_START = "ignore_start"


def _get_table_for_column_nesting(
    df,
    sep=".-?",
    to_level=0,
    ignore_pattern=r".*_[A-Z]",
    separator_in_name: Literal["before", "after", "drop"] = "before",
    allow_singleton_nesting=False,
    nesting_strategy: NestingStrategy = NestingStrategy.IGNORE_START,
):
    if nesting_strategy == NestingStrategy.IGNORE_START:
        SEP = re.escape(sep)
        PATTERN = rf"(?:{ignore_pattern})?([^\n{SEP}]+)?([{SEP}])?"
        # PATTERN = rf"(?:{ignore_pattern})?([^\n({SEP})]+)?([({SEP})])?"
        column_components = (
            pl.Series("columns", df.columns)
            .to_frame()
            .with_columns(
                pl.col("columns").str.extract_all(PATTERN).alias("components")
            )
        )
    elif nesting_strategy == NestingStrategy.SPLIT:
        column_components = (
            pl.Series("columns", df.columns)
            .to_frame()
            .with_columns(
                pl.col("columns").str.split(sep, inclusive=True).alias("components")
            )
            .with_columns(pl.col("components").list.slice(0, pl.len() - 1))
        )

    LEN_SEP = len(sep) if nesting_strategy == "split" else 1
    long_table = (
        column_components.with_row_index()
        .explode("components")
        .with_columns(
            pl.col("components"),
            pl.int_range(0, pl.len()).over("index").alias("level"),
        )
        .with_columns(
            (pl.col("level") != pl.col("level").max()).over("index").alias("must_split")
        )
        .with_columns(
            pl.when(pl.col("must_split"))
            .then(
                pl.concat_list(
                    pl.col("components").str.slice(
                        0, pl.col("components").str.len_chars() - LEN_SEP
                    ),
                    pl.col("components").str.slice(
                        pl.col("components").str.len_chars() - LEN_SEP
                    ),
                )
            )
            .otherwise(pl.concat_list(pl.col("components")))
            .alias("components_with_sep")
        )
        .explode("components_with_sep")
        .with_columns(
            (pl.col("level").diff(n=-1) == -1).fill_null(False).alias("is_separator")
        )
        .with_columns(
            (pl.col("level") + pl.col("is_separator")).alias("level_sep_after")
        )
        .select(
            pl.col("index"),
            pl.col("columns"),
            pl.col("components"),
            pl.col("level"),
            pl.col("level_sep_after"),
            pl.col("components_with_sep"),
            pl.col("is_separator"),
        )
        .group_by(
            "index",
            "columns",
            *(
                (
                    pl.col("level_sep_after").alias("level")
                    if separator_in_name == "after"
                    else "level"
                ),
                "is_separator",
            ),
            maintain_order=True,
        )
        .agg(pl.col("components_with_sep"))
        .with_columns(pl.col("components_with_sep").list.join(""))
    )
    return long_table


def nest_selector(
    df,
    sep=".-?",
    to_level=0,
    method=NestingStrategy.SPLIT,
    ignore_pattern=r".*_[A-Z]",
    separator_in_name: Literal["before", "after", "drop"] = "after",
    allow_singleton_nesting=False,
    column_selector=cs.all(),
):
    columns_to_nest = cs.expand_selector(df, column_selector)
    columns_to_ignore = cs.expand_selector(df, ~column_selector)

    columns_to_nest_index = [df.columns.index(c) for c in columns_to_nest]
    columns_to_ignore_index = [df.columns.index(c) for c in columns_to_ignore]

    long_table = _get_table_for_column_nesting(
        df,
        sep=sep,
        to_level=to_level,
        ignore_pattern=ignore_pattern,
        separator_in_name=separator_in_name,
    ).filter(pl.col("index").is_in(columns_to_nest_index))
    wide_table = long_table.pivot(
        index=["index", "columns"],
        values="components_with_sep",
        columns=["level", "is_separator"],
    )
    max_level = long_table["level"].max()
    current_level = max_level
    selectors = []

    while current_level > to_level:
        group_columns = [cs.matches(r"^\{" + str(c)) for c in range(current_level)]
        field_columns = [cs.matches(r"^\{" + str(current_level))]
        actual_group_columns = reduce(
            add, [cs.expand_selector(wide_table, e) for e in group_columns]
        )
        # Include the separator in the grouping columns
        # (only relevant for `separator_in_name='after'`)
        group_separator = cs.expand_selector(
            wide_table, cs.matches(r"^\{" + str(current_level) + ",true}$")
        )
        extra_group_columns = ["group_sep"] if len(group_separator) == 1 else []

        agg = (
            wide_table.with_columns(pl.col(group_separator).alias("group_sep"))
            .group_by(group_columns + extra_group_columns, maintain_order=True)
            .agg(
                pl.col("columns").alias("old_columns"),
                pl.concat_str(field_columns).alias("field_names"),
                pl.col("index").min(),
            )
            .with_columns(
                pl.concat_str(group_columns, ignore_nulls=True).alias("columns")
            )
            .with_columns(
                (pl.col("field_names").list.len() <= 1).alias("singleton_struct")
            )
        )

        struct_expression_table = agg.select(
            ["old_columns", "field_names", "columns", "singleton_struct", "index"]
        )

        selector = []
        for (
            old_columns,
            field_names,
            new_column,
            is_singleton,
            index,
        ) in struct_expression_table.rows():
            if is_singleton and (field_names[0] is None or not allow_singleton_nesting):
                selector_col = (index, pl.col(old_columns))

            else:
                selector_col = (
                    index,
                    (
                        pl.struct(old_columns)
                        .struct.rename_fields(field_names)
                        .alias(new_column)
                    ),
                )
            selector.append(selector_col)
        # add passthrough columns
        selector.extend(
            [
                (index, cs.by_name(col_name))
                for index, col_name in zip(columns_to_ignore_index, columns_to_ignore)
            ]
        )
        # if
        selectors.append([e[1] for e in sorted(selector)])

        if not allow_singleton_nesting:
            agg = agg.with_columns(
                pl.when(pl.col("singleton_struct"))
                .then(
                    pl.concat_str(
                        pl.col(actual_group_columns[-1]),
                        pl.col("field_names").list.join(""),
                    ),
                )
                .otherwise(pl.col(actual_group_columns[-1])),
                pl.when(pl.col("singleton_struct"))
                .then(
                    pl.concat_str(
                        pl.col("columns"), pl.col("field_names").list.join("")
                    ),
                )
                .otherwise(pl.col("columns")),
            )

        wide_table = agg.select("index", "columns", *group_columns)
        current_level -= 1
    return selectors


def nest(
    df,
    sep=".",
    to_level=0,
    column_selector=cs.all(),
    ignore_pattern=r".*_[A-Z]",
    separator_in_name="after",
    allow_singleton_nesting=False,
):
    for ns in nest_selector(
        df,
        sep=sep,
        to_level=to_level,
        ignore_pattern=ignore_pattern,
        separator_in_name=separator_in_name,
        allow_singleton_nesting=allow_singleton_nesting,
        column_selector=column_selector,
    ):
        df = df.select(ns)
    return df


def dtype_depth(dtype):
    if isinstance(dtype, pl.Field):
        dtype = dtype.dtype

    if dtype.is_nested():
        if hasattr(dtype, "fields"):
            return 1 + (max(map(dtype_depth, dtype.fields)) if dtype else 0)
    return 0


def unnest(df, to_level=-1, sep="."):
    max_depth = max(map(dtype_depth, df.dtypes))
    if to_level < 0:
        to_level = max_depth + to_level + 1
    for _ in range(to_level):
        df = df.pipe(unnest_all_structs, sep=sep)
    return df


def nest_struct_nth_pos(df, n=0, sep=".", from_last=False):
    if not from_last:
        select_nth = pl.col("parts").list.get(n).alias("inside")
        select_pre = pl.col("parts").list.slice(0, n).alias("pre")
        select_post = pl.col("parts").list.slice(n + 1).alias("post")
        join_parts = (
            pl.concat_list(select_pre, select_post).list.join(sep).alias("outside")
        )
    else:
        select_nth = pl.col("parts").list.reverse().list.get(n).alias("inside")
        select_pre = pl.col("parts").list.reverse().list.slice(0, n).alias("pre")
        select_post = pl.col("parts").list.reverse().list.slice(n + 1).alias("post")
        join_parts = (
            pl.concat_list(select_pre, select_post)
            .list.reverse()
            .list.join(sep)
            .alias("outside")
        )

    struct_columns = (
        (
            pl.Series("columns", df.columns)
            .to_frame()
            .with_columns(pl.col("columns").str.split(".").alias("parts"))
        )
        .select(
            pl.col("columns"),
            select_nth,
            join_parts,
        )
        .with_columns(
            pl.when(pl.col("outside") == "")
            .then(pl.col("inside"))
            .otherwise(pl.col("outside"))
            .alias("outside")
        )
        .group_by("outside", maintain_order=True)
        .agg(pl.all())
        .select("outside", "columns", "inside")
    )

    command = []
    for outside, columns, inside in struct_columns.select(
        ["outside", "columns", "inside"]
    ).rows():
        if len(columns) <= 1:
            command.append(pl.col(columns))
        else:
            command.append(
                pl.struct(columns).struct.rename_fields(inside).alias(outside)
            )

    return df.select(command)


def get_correlation_by_acquisition_map(df: pl.DataFrame) -> dict:
    """Get the correlation by acquisition map from the dataframe.

    Uses regex to match correlation column names in the format
    'Channel1_Index1|Channel2_Index2_Metric'.

    Args:
        df: DataFrame containing correlation columns

    Returns:
        Dictionary mapping correlation column names to acquisition numbers
    """
    # Pattern to match correlation column names
    pattern = r"([A-Za-z0-9]+)_(\d+)\|([A-Za-z0-9]+)_(\d+)_([A-Za-z0-9]+)"

    corr_map = {}
    correlation_cols = [col for col in df.columns if re.match(pattern, col)]

    for col in correlation_cols:
        match = re.match(pattern, col)
        if match:
            _, idx1, _, _, _ = match.groups()
            acquisition = int(idx1)
            corr_map[col] = acquisition

    return corr_map


def stack_correlation_metric_by_acquisition(
    df_corr: pl.DataFrame,
    corr_map: dict[str, int],
):
    return (
        df_corr.rename(valmap(str, corr_map))
        .unpivot(
            index=["ROI", "object", "label", "well"],
            on=list(map(str, corr_map.values())),
            variable_name="variable",
            value_name="value",
        )
        .with_columns(
            [
                pl.col("variable").cast(pl.UInt16).alias("acquisition"),
                pl.col("value").alias("alignmentScore"),
            ]
        )
        .drop(["variable", "value"])
    )


def plot_channel_t_decay_models(
    df_mean_per_embryo, output_dir, y_name="Mean_CycleMeanNorm", loss="linear", n_cols=4
):
    """Plot intensity decay models for all channels.

    Parameters:
    -----------
    df_mean_per_embryo : polars.DataFrame
        DataFrame containing channel intensity data
    output_dir : str
        Directory to save the output figure
    y_name : str
        Column name for intensity values
    loss : str
        Loss function for model fitting ('linear' or 'huber')
    n_cols : int
        Number of columns in the subplot grid
    """
    # Configuration
    mdl_types = [Exp, ExpNoOffset, Linear, LogLinear]
    mdl_colors = ["#dd966d", "#6ddd96", "#966ddd"]
    scatter_color = "#8e8e8e"

    # Convert to pandas for easier plotting
    to_plot = df_mean_per_embryo.to_pandas()

    # Extract and sort channels by their numeric index
    channels = next(iter(df_mean_per_embryo.select(pl.col("channel")).unique()))
    channels = sorted(channels, key=lambda x: int(x.rsplit("_", 1)[1]))

    # Create subplot grid
    n_rows = math.ceil(len(channels) / n_cols)
    fig, axs = plt.subplots(
        n_rows, n_cols, sharex=True, sharey=False, figsize=(15, n_rows * 3), dpi=200
    )

    # Flatten axes array for easy iteration
    axs_flat = axs.flatten() if hasattr(axs, "flatten") else axs

    # Track models for each channel
    models = {}
    for i, (channel, ax) in enumerate(zip(channels, axs.flatten())):
        models[channel] = {}
        df_channel = to_plot[to_plot.channel == channel]
        X = df_channel[["timeDeltaMinutes"]]
        y = df_channel[y_name]
        X_fit = np.linspace(
            df_channel["timeDeltaMinutes"].min(),
            df_channel["timeDeltaMinutes"].max(),
            200,
        ).reshape(-1, 1)

        # Handle potentially 1D array of axes when n_rows=1
        if n_rows == 1:
            ax = axs[i % n_cols]
        else:
            ax = axs[i // n_cols, i % n_cols]

        plt.sca(ax)
        if i % n_cols == 0:
            plt.ylabel("Mean Intensity [AU]")
        if i // n_cols == (n_rows - 1):
            plt.xlabel("Acquisition Time [min]")
        plt.title(f"{channel}")
        sns.scatterplot(
            x=X.values.flatten(),
            y=y.values,
            color=scatter_color,
            s=10,
            label=f"{channel} Mean",
        )

        plt.ylim(bottom=0)

        for mdl_type, mdl_color in zip(mdl_types, mdl_colors):
            try:
                mdl = mdl_type(loss=loss).fit(X, y)
                y_mdl = mdl.predict(X_fit)
                plt.plot(
                    X_fit, y_mdl, color=mdl_color, label=mdl_type.__name__, linewidth=2
                )
                models[channel][mdl_type.__name__] = mdl
            except RuntimeError:
                print(f"{mdl_type.__name__} not converged on {channel}")
                models[channel][mdl_type.__name__] = None
        plt.legend()

    # Hide empty subplots
    for i in range(len(channels), n_rows * n_cols):
        row, col = i // n_cols, i % n_cols
        if n_rows == 1:
            axs[col].set_visible(False)
        else:
            axs[row, col].set_visible(False)

        # Hide empty subplots
        for i in range(len(channels), len(axs_flat)):
            axs_flat[i].set_visible(False)

    # Save figure
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/time_decay_models.png", dpi=300, bbox_inches="tight")

    return models, fig


def replace_channel_separators(s: pl.Series) -> pl.Series:
    return s.str.replace_all(r"(\w)-(\d)", "$1.$2").str.replace_all(
        r"(\w+.\d+)-(\w+.\d+)", "$1|$2"
    )


def replace_channel_separators_in_columns(df: pl.DataFrame) -> pl.DataFrame:
    rename_map = dict(
        zip(
            *pl.Series("columns", df.columns)
            .to_frame()
            .with_columns(
                pl.col("columns").map(replace_channel_separators).alias("renamed")
            )
            .to_dict(as_series=False)
            .values()
        )
    )
    return df.rename(rename_map)


def join(
    dfs: Sequence[pl.DataFrame],
    on: str | Sequence[str],
    how: JoinStrategy = "outer_coalesce",
) -> pl.DataFrame:
    if len(dfs) == 0:
        return pl.DataFrame()
    df = dfs[0]
    for df_other in dfs[1:]:
        df = df.join(df_other, on=on, how=how)
    return df


def _split_feature_name(columns: list[str], sep="_") -> dict[str, dict[str, str]]:
    res = defaultdict(dict)
    for column in columns:
        res[sep.join(column.split(sep)[:-1])][column] = column.split(sep)[-1]
    return dict(res)


def split_and_melt_column_name(
    df: pl.DataFrame,
    sep: str = "_",
    pattern_before_sep: str | None = None,
    pattern_after_sep: str | None = None,
    new_column_name: str = "resources",
    index: "SelectorType | Iterable[str] | None" = None,
    return_list: bool = False,
):
    if pattern_before_sep is not None or pattern_after_sep is not None:
        raise NotImplementedError("upsi")
    return stack_column_name_to_column(
        df, sep=sep, column_name=new_column_name, index=index, return_list=return_list
    )


def split_and_melt_column_name_expr(
    column: str,
    sep: str = "_",
    pattern_before: str | None = None,
    pattern_after: str | None = None,
    column_before_split: bool = True,
    new_column_name: str = "column",
    drop_non_matching: bool = False,
) -> pl.Expr:
    if pattern_before is None:
        pattern_before = f"(.*[^{sep}])?"
    if pattern_after is None:
        pattern_after = f"([^{sep}].*)?"
    pattern = re.compile(f"{pattern_before}{sep}{pattern_after}")

    mo = pattern.match(column)
    if mo is None:
        return pl.col(column)

    values_column, column_name = mo.groups()
    if not column_before_split:
        values_column, column_name = column_name, values_column

    return pl.struct(
        pl.lit(values_column).alias(new_column_name), pl.col(column).alias(column_name)
    ).alias(column)


def split_and_melt_column_names_on(
    df: AnyFrameT,
    sep: str = "_",
    pattern_before: str | None = None,
    pattern_after: str | None = None,
    column_before_split: bool = False,
    new_column_name: str = "column",
    index_columns: "str | Iterable[str] | SelectorType" = sel.index
    | cs.starts_with("idx.")
    | cs.starts_with("fidx."),
) -> AnyFrameT:
    # df = df.lazy()
    if pattern_before is None:
        pattern_before = f".*[^{sep}]"
    if pattern_after is None:
        pattern_after = f"[^{sep}].*"
    pattern = f"^({pattern_before}){sep}({pattern_after})$"

    split_columns = (
        (
            pl.Series("columns", cs.expand_selector(df, ~index_columns))
            .to_frame()
            .with_columns(
                [
                    pl.col("columns")
                    .str.extract_groups(pattern)
                    .alias("parts")
                    .struct.rename_fields(["before", "after"])
                ]
            )
            .unnest("parts")
        )
        # )
        .drop_nulls()
        .group_by("after" if column_before_split else "before", maintain_order=True)
        .agg(pl.all())
        .rows()
    )
    out_dfs = []
    for column_value, old_names, new_names in split_columns:
        out_dfs.append(
            df.select(
                index_columns,
                pl.lit(column_value).alias(new_column_name),
                *[pl.col(o).alias(n) for o, n in zip(old_names, new_names)],
            )
        )
    # return out_dfs
    if len(out_dfs) == 0:
        return df
    return pl.concat(out_dfs, how="diagonal")


def is_selector(s: Any) -> TypeGuard[SelectorType]:
    return cs.is_selector(s)


def stack_column_name_to_column(
    df: pl.DataFrame,
    sep: str = "_",
    column_name: str = "resources",
    index: "SelectorType | tuple[str, ...] | None" = None,
    return_list: bool = False,
) -> pl.DataFrame:
    if index is None:
        index = sel.index
    if is_selector(index):
        index = cs.expand_selector(df, index)
    split = _split_feature_name([e for e in df.columns if e not in index], sep=sep)

    dfs = []
    for split_name, rename_map in split.items():
        df_sub = df.select(
            [pl.col(e) for e in index] + [pl.col(e) for e in rename_map.keys()]
        )
        dfs.append(
            df_sub.with_columns(pl.lit(split_name).alias(column_name))
            .rename(rename_map)
            .drop_nulls(list(rename_map.values()))
            .select(
                pl.col(*list(index), column_name),
                pl.exclude(*list(index), column_name),
            )
        )
    if return_list:
        return dfs
    return pl.concat(dfs, how="diagonal")


def unstack_column_to_column_name(
    df: pl.DataFrame,
    sep: str = "_",
    column_name: str = "resources",
    index: tuple[str, ...] = ("ROI", "object", "label"),
) -> pl.DataFrame:
    channel_names = df.select(column_name).unique()[column_name].to_list()
    dfs = []
    for channel_name in channel_names:
        dfs.append(
            df.filter(pl.col(column_name) == channel_name).select(
                [
                    *index,
                    pl.exclude([*index, column_name]).prefix(f"{channel_name}{sep}"),
                ]
            )
        )
    return join(dfs, on=list(index), how="outer_coalesce")


def split_column(
    df: pl.DataFrame,
    split_column_name="channel",
    out_column_names=("stain", "acquisition"),
    sep=".",
    index=("ROI", "object", "label"),
) -> pl.DataFrame:
    return df.with_columns(
        [
            pl.col(split_column_name)
            .str.split_exact(by=sep, n=len(out_column_names) - 1)
            .struct.field(f"field_{i}")
            .alias(out_column_names[i])
            for i in range(len(out_column_names))
        ]
    ).select(
        [
            pl.col([*index, split_column_name, *out_column_names]),
            pl.exclude([*index, split_column_name, *out_column_names]),
        ]
    )


def split_channel_column(
    df: pl.DataFrame,
    index=("ROI", "object", "label"),
) -> pl.DataFrame:
    return split_column(
        df=df,
        split_column_name="channel",
        out_column_names=("stain", "acquisition"),
        sep="_",
        index=index,
    )


def split_channel_pair_column(
    df: pl.DataFrame,
    index: tuple[str, ...] = ("ROI", "object", "label"),
    force_reference_column: Literal["auto"] | str | None = None,
) -> pl.DataFrame:
    df_split = split_column(
        df=df,
        split_column_name="channel_pair",
        out_column_names=("channel", "channel_ref"),
        sep="|",
        index=index,
    )
    if force_reference_column is not None:
        df_split = df_split.with_columns(
            [
                pl.when(pl.col("channel") == force_reference_column)
                .then(pl.col("channel_ref"))
                .otherwise(pl.col("channel"))
                .alias("channel"),
                pl.when(pl.col("channel_ref") == force_reference_column)
                .then(pl.col("channel_ref"))
                .otherwise(pl.col("channel"))
                .alias("channel_ref"),
            ]
        ).pipe(split_channel_column, index=index)
    return df_split


def select_numeric_and_nested(df: pl.DataFrame) -> pl.DataFrame:
    return df.select(
        [
            e
            for e, dtype in zip(df.columns, df.dtypes)
            if dtype in [pl.Struct, pl.List, *pl.NUMERIC_DTYPES]
        ]
    )


# def apply_colormap(s: pl.Series, cmap=cc.m_glasbey) -> pl.Series:
#     unique_values = s.unique().sort().to_list()
#     color_map = {v: cmap(unique_values.index(v)) for v in unique_values}
#     return s.map_dict(color_map)


# def plot_df_nulls(
#     df: pl.DataFrame,
#     index: tuple[str, ...] = ("roi", "object", "label"),
#     row_color_columns: tuple[str, ...] | None = None,
#     subsample_by: int | None = None,
#     title: str | None = None,
#     **kwargs,
# ):
#     df = df.fill_nan(None)
#     if subsample_by is not None:
#         df_samp = df.gather_every(subsample_by)
#     else:
#         df_samp = df

#     # pdf_index = df.select(pl.col(e) for e in index).to_pandas()
#     pdf_cats = (
#         None
#         if row_color_columns is None
#         else df_samp.select(
#             pl.col(e).map(apply_colormap) for e in row_color_columns
#         ).to_pandas()
#     )

#     feature_columns = sorted(
#         list(set(df_samp.pipe(select_numeric_and_nested).columns).difference(index)),
#         key=df_samp.columns.index,
#     )
#     pdf_features_is_null = (
#         df_samp.select(feature_columns).select(pl.all().is_null()).to_pandas()
#     )
#     nan_percentage = pdf_features_is_null.sum().sum() / pdf_features_is_null.size

#     default_params = dict(
#         figsize=(5, 6),
#         row_cluster=False,
#         col_cluster=False,
#         cmap=sns.color_palette(list("gr"), as_cmap=True),
#         cbar_pos=(0.02, 0.5, 0.1, 0.05),
#         cbar_kws={"boundaries": [0, 0.5, 1], "orientation": "horizontal"},
#         row_colors=pdf_cats,
#         vmin=0,
#         vmax=1,
#     )

#     g = sns.clustermap(
#         pdf_features_is_null,
#         **{
#             **default_params,
#             **kwargs,
#         },
#     )

#     if g.ax_cbar:
#         g.ax_cbar.set_xticks([0.25, 0.75])
#         g.ax_cbar.xaxis.tick_top()
#         g.ax_cbar.set_xticklabels(
#             ["not null", f"null ({nan_percentage:.2f})"],
#             rotation=90,  # , verticalalignment="center"
#         )

#     if g.ax_row_colors:
#         g.ax_row_colors.xaxis.tick_top()
#         xticklabels = g.ax_row_colors.get_xticklabels()
#         g.ax_row_colors.set_xticklabels(xticklabels, rotation=90)

#     g.ax_heatmap.xaxis.tick_top()
#     xticklabels = g.ax_heatmap.get_xticklabels()
#     g.ax_heatmap.set_xticklabels(xticklabels, rotation=90)
#     g.ax_heatmap.set_yticks([])
#     g.ax_heatmap.set_xlabel(f"{pdf_features_is_null.shape[1]}")
#     g.ax_heatmap.set_ylabel(
#         f"{pdf_features_is_null.shape[0]:,}{f' out of {df.shape[0]:,}'}"
#     )

#     return g

# def pipe_df_nulls(df, **kwargs):
#     plot_df_nulls(df, **kwargs)
#     return df


def plot_null_columns(
    df: pl.DataFrame,
    index: SelectorType = sel.index | cs.starts_with("idx.") | cs.starts_with("fidx."),
    sort=True,
    drop_zero_columns=False,
    ax=None,
    max_labeled_features: int = 30,
):
    import matplotlib.pyplot as plt

    null_counts_table = column_null_count(
        df, drop_zero_columns=drop_zero_columns, sort=sort
    )
    values = null_counts_table["null_percentage"]

    if ax is None:
        _, ax = plt.subplots(figsize=(2, 6))

    ax.plot(values, np.arange(len(values)))
    if len(values) > max_labeled_features:
        take_every_nth = round(len(values) / max_labeled_features)
        ax.set_yticks(np.arange(len(values)), minor=True)
    else:
        take_every_nth = 1
    ax.set_yticks(np.arange(len(values))[::take_every_nth])
    ax.set_yticklabels(null_counts_table["feature"][::take_every_nth])
    ax.set_xlabel("null percentage")
    ax.set_ylabel(f"feature ({len(values)})")


def null_percentage(
    df: pl.DataFrame, index: tuple[str, ...] = ("ROI", "object", "label")
) -> float:
    df_no_index = df.select(pl.exclude(index))
    return (
        df_no_index.select(pl.all().null_count()).sum(axis=1).item()
        / df_no_index.select(pl.all().count()).sum(axis=1).item()
    )
