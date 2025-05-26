"""Functions to mark outliers in a DataFrame based on specified ranges."""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import zip_longest
from typing import Literal, Protocol, runtime_checkable

import numpy as np
import polars as pl

from abbott_features.intensity_normalization.plot_features import iqr_range
from abbott_features.intensity_normalization.polars_utils import AnyFrameT

logger = logging.getLogger(__name__)


def identity(x):
    return x


TRANSFORMS = {
    "log10": np.log10,
    "log2": np.log2,
    "log": np.log,
    "identity": identity,
}


@runtime_checkable
class Outlier(Protocol):
    feature: str
    transform: Literal["log10", "log2", "log", "identity"]

    def fit(self, df: pl.DataFrame) -> "Outlier": ...

    @property
    def lower(self) -> str | None: ...

    @property
    def upper(self) -> str | None: ...

    def indicate_lower_outlier(self, df: pl.DataFrame) -> pl.Series:
        self.fit(df)
        col_expression = pl.col(self.feature).map_elements(
            TRANSFORMS[self.transform], return_dtype=pl.Float32
        )
        if self.lower is None:
            return df.select(col_expression.is_null()).to_series()
        return df.select(col_expression.lt(self.lower)).to_series()

    def indicate_upper_outlier(self, df: pl.DataFrame) -> pl.Series:
        self.fit(df)
        col_expression = pl.col(self.feature).map_elements(
            TRANSFORMS[self.transform], return_dtype=pl.Float32
        )
        if self.upper is None:
            return df.select(col_expression.is_null()).to_series()
        return df.select(col_expression.gt(self.upper)).to_series()

    def indicate_outlier(self, df: pl.DataFrame) -> pl.Series:
        return (
            self.indicate_lower_outlier(df) | self.indicate_upper_outlier(df)
        ).to_frame()

    def __str__(self) -> str:
        return self.__repr__()


@dataclass
class RngOutlier(Outlier):
    feature: str
    lower: float | None = None
    upper: float | None = None
    transform: Literal["log10", "log2", "log", "identity"] = "identity"

    def fit(self, df):
        return self


def mark_outliers(df: AnyFrameT, outliers: list[Outlier], verbose=True) -> pl.Series:
    is_outlier = pl.concat(
        [outlier.indicate_outlier(df) for outlier in outliers],
        how="horizontal",
    ).with_columns(
        pl.all_horizontal(pl.all()).alias("All Outliers"),
        pl.any_horizontal(pl.all()).alias("Any Outliers"),
    )
    if verbose:
        is_below = pl.DataFrame(
            [outlier.indicate_lower_outlier(df) for outlier in outliers],
        ).with_columns(
            pl.all_horizontal(pl.all()).alias("All Outliers"),
            pl.any_horizontal(pl.all()).alias("Any Outliers"),
        )
        is_above = pl.DataFrame(
            [outlier.indicate_upper_outlier(df) for outlier in outliers],
        ).with_columns(
            pl.all_horizontal(pl.all()).alias("All Outliers"),
            pl.any_horizontal(pl.all()).alias("Any Outliers"),
        )
    else:
        is_below = is_above = None

    outlier_summary(is_outlier, is_below, is_above)
    return is_outlier.select(pl.col("Any Outliers")).to_series()


def outlier_summary(
    df_outliers: pl.DataFrame, is_below=None, is_above=None
) -> pl.DataFrame:
    if hasattr(df_outliers, "to_frame"):
        df_outliers = df_outliers.to_frame()
    n_total = df_outliers.height
    for c in df_outliers.columns:
        n_outliers = df_outliers[c].sum()
        logger.info(f"{c}")
        logger.info(
            f"{n_outliers:>8,}/{n_total:<8,}={n_outliers/n_total:>7.2%} "
            "outliers or null"
        )
        if is_below is not None:
            n_below = is_below[c].sum()
            logger.info(f"{n_below:>8,}/{n_total:<8,}={n_below/n_total:>7.2%} below")
        if is_above is not None:
            n_above = is_above[c].sum()
            logger.info(f"{n_above:>8,}/{n_total:<8,}={n_above/n_total:>7.2%} above")


def drop_outliers(df, outliers: Sequence[Outlier], verbose=True, plot_results=False):
    df_outliers = mark_outliers(df, outliers, verbose=verbose)
    if verbose:
        logger.info(f"{' Removing outliers ':=^50}")
        outlier_summary(df_outliers)
    df_out = df.filter(~df_outliers)
    if plot_results:
        results_plot(df, df_out, outliers)
    return df_out


def remove_outliers(df, outliers: Sequence[Outlier], verbose=True, plot_results=False):
    return drop_outliers(
        df=df, outliers=outliers, verbose=verbose, plot_results=plot_results
    )


def clip_outliers(df, outliers: Sequence[Outlier], verbose=True, plot_results=False):
    if verbose:
        df_outliers = mark_outliers(df, outliers, verbose=verbose)
        logger.info(f"{' Clipping values to outlier ranges ':=^50}")
        outlier_summary(df_outliers)
    df_out = df.clone()
    for outlier in outliers:
        outlier.fit(df)
        if outlier.lower is not None:
            df_out = df_out.with_columns(
                pl.col(outlier.feature).clip_min(outlier.lower)
            )
        if outlier.upper is not None:
            df_out = df_out.with_columns(
                pl.col(outlier.feature).clip_max(outlier.upper)
            )
    if plot_results:
        results_plot(df, df_out, outliers)
    return df_out


def results_plot(df, df_out, outliers: Sequence[Outlier], ncols=4, figsize=None):
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    last_axes = False
    nrows = int(np.ceil(len(outliers) / ncols))
    figsize = figsize or (ncols * 4, nrows * 3)
    fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=figsize, dpi=300)
    old_ax = axs[0]
    for outlier, ax in zip_longest(outliers, axs.flatten()):
        if outlier is None:
            ax.remove()
            if last_axes is False:
                old_ax.legend()
                continue
        sns.kdeplot(df[outlier.feature], ax=ax, label="before")
        sns.kdeplot(df_out[outlier.feature], ax=ax, label="after")
        ax.set_xlim(*iqr_range(df[outlier.feature], q_lower=0.1, q_upper=0.9, r=3))
        ax.set_title(outlier.feature)
        if outlier.lower is not None:
            ax.axvline(x=outlier.lower, color="k", linestyle="dotted")
        if outlier.upper is not None:
            ax.axvline(x=outlier.upper, color="k", linestyle="dashed")

        old_ax = ax
