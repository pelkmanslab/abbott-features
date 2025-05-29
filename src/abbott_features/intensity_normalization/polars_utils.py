"""Utils functions for polars DataFrames."""

import logging
import math
import os
import re
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, TypeGuard, TypeVar

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import polars.selectors as cs
import seaborn as sns
from polars.type_aliases import JoinStrategy, SelectorType
from toolz.dicttoolz import valmap

from abbott_features.intensity_normalization.models import (
    Exp,
    ExpNoOffset,
    Linear,
    LogLinear,
)
from abbott_features.intensity_normalization.polars_selector import sel

logger = logging.Logger(__name__)

AnyFrameT = TypeVar("AnyFrameT", pl.DataFrame, pl.LazyFrame)


def lower_upper_iqr(series, q_lower=0.25, q_upper=0.75):
    ql = series.quantile(q_lower)
    qu = series.quantile(q_upper)
    iqr = qu - ql
    return ql, qu, iqr


def _split_feature_name(columns: list[str], sep="_") -> dict[str, dict[str, str]]:
    res = defaultdict(dict)
    for column in columns:
        res[sep.join(column.split(sep)[:-1])][column] = column.split(sep)[-1]
    return dict(res)


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


def is_selector(s: Any) -> TypeGuard[SelectorType]:
    return cs.is_selector(s)
