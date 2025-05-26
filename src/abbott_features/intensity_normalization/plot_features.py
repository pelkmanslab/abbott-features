"""Functions to visualize features."""

from itertools import zip_longest

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import polars.selectors as cs
import seaborn as sns

from abbott_features.intensity_normalization.polars_preprocessing import META_COLUMNS
from abbott_features.intensity_normalization.polars_selector import sel
from abbott_features.intensity_normalization.polars_utils import (
    split_channel_column,
    split_channel_pair_column,
    stack_column_name_to_column,
    unnest_all_structs,
)


def lower_upper_iqr(series, q_lower=0.25, q_upper=0.75):
    ql = series.quantile(q_lower)
    qu = series.quantile(q_upper)
    iqr = qu - ql
    return ql, qu, iqr


def iqr_range(
    series, q_lower: float | None = 0.25, q_upper: float | None = 0.75, r: float = 1.5
):
    if q_lower is None:
        q_lower = 0.0
        ql, qu, iqr = lower_upper_iqr(series, q_lower=q_lower, q_upper=q_upper)
        return ql, ql + r * iqr
    if q_upper is None:
        q_upper = 1.0
        ql, qu, iqr = lower_upper_iqr(series, q_lower=q_lower, q_upper=q_upper)
        return qu - r * iqr, qu

    ql, qu, iqr = lower_upper_iqr(series, q_lower=q_lower, q_upper=q_upper)
    return ql - (r - 1) / 2 * iqr, qu + (r - 1) / 2 * iqr


def plot_correlation(corr, figsize=(10, 10)):
    ax = sns.clustermap(corr, vmin=-1, vmax=1, center=0, cmap="RdBu", figsize=figsize)
    return ax


def plot_features_by_embryo(
    current_features, hue=None, q_lower=0.0, q_upper=1.0, q_range=1, sharex=False
):
    if isinstance(current_features, pl.DataFrame):
        current_features = current_features.to_pandas()
    if hue is not None:
        current_features = current_features.set_index(hue)
    plot_features = [e for e in current_features.columns if e != hue]
    n_features = len(plot_features)
    n_cols = 4
    n_rows = int(np.ceil(n_features / n_cols))

    fig, axs = plt.subplots(
        n_rows, n_cols, figsize=(20, 5 * n_rows), dpi=300, sharex=sharex
    )
    axs = axs.flatten()

    for i, (ax, feature) in enumerate(zip_longest(axs, plot_features)):
        plt.sca(ax)
        if feature is None:
            ax.remove()
            continue
        rng = iqr_range(
            current_features[feature], q_lower=q_lower, q_upper=q_upper, r=q_range
        )
        if hue is None:
            sns.histplot(
                data=current_features.clip(*rng).reset_index(),
                x=feature,
                hue=hue,
                legend=False,
                common_norm=False,
                stat="density",
            )
        sns.kdeplot(
            data=current_features.clip(*rng).reset_index(),
            x=feature,
            hue=hue,
            legend=i == 0,
            common_norm=False,
            warn_singular=False,
            cut=0,
            c="k",
        )


def mm_to_inch(mm):
    return mm / 25.4


def plot_embryos_by_age(
    df_meta, figsize=(mm_to_inch(100.5), mm_to_inch(68)), dpi=300, **scatterplot_kwrags
):
    Y_FEATURE = "log2_nucleiRaw3_Count"
    to_plot = df_meta.sort(Y_FEATURE).with_row_count("rank").to_pandas()

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    plt.plot(to_plot["rank"], to_plot[Y_FEATURE], color="k", lw=0.8)

    default_kwargs = {
        "data": to_plot,
        "x": "rank",
        "y": Y_FEATURE,
        "hue": "cycle",
        "palette": "viridis",
        "zorder": 10,
        "s": 6,
    }
    sns.scatterplot(
        **{
            **default_kwargs,
            **scatterplot_kwrags,
        }
    )
    sns.move_legend(ax, loc="best", markerscale=2.0)
    plt.xlabel(f"embryo (n={len(to_plot)})")
    plt.ylabel("$log_2$(nuclei count)")
    plt.xticks([])
    plt.grid(linestyle="--")
    plt.suptitle("Embryo nuclei counts", x=0.0, ha="left")


def plot_correlation_per_embryo_w_errorbars(df_nuc, df_meta):
    corr_feature = "PearsonR"
    channel_pairs = ["DAPI.0|DAPI.1", "DAPI.1|DAPI.2", "DAPI.1|DAPI.3"]
    x = "nucleiRaw3_Count"
    y = f"{corr_feature}_Mean"
    yerr_upp = "yerr_upp"
    yerr_low = "yerr_low"

    to_plot = (
        df_nuc.select(sel.index, cs.matches(corr_feature).fill_nan(pl.lit(0.0)))
        .pipe(stack_column_name_to_column, column_name="channel_pair", index=sel.index)
        .groupby(["roi", "channel_pair"])
        .agg(
            pl.col(corr_feature).mean().suffix("_Mean"),
            pl.col(corr_feature).quantile(0.9).suffix("_Q.9"),
            pl.col(corr_feature).quantile(0.1).suffix("_Q.1"),
        )
        .with_columns(
            (pl.col(f"{corr_feature}_Mean") - pl.col(f"{corr_feature}_Q.9"))
            .abs()
            .alias("yerr_upp"),
            (pl.col(f"{corr_feature}_Mean") - pl.col(f"{corr_feature}_Q.1"))
            .abs()
            .alias("yerr_low"),
        )
        .join(df_meta, on="roi")
    )

    fig, axs = plt.subplots(
        3, 1, dpi=300, figsize=(12, 12), layout="constrained", sharey=True
    )

    for ax, channel_pair in zip(axs.flatten(), channel_pairs):
        ax: plt.Axes
        df_single = to_plot.filter(pl.col("channel_pair") == channel_pair)
        ax.scatter(df_single[x], df_single[y], s=4, zorder=100)
        # ax.scatter(df_single[x], df_single['PearsonR_Q.9'], s=4)
        # ax.scatter(df_single[x], df_single['PearsonR_Q.1'], s=4)
        ax.errorbar(
            df_single[x],
            df_single[y],
            yerr=df_single.select([yerr_low, yerr_upp]).to_numpy().T,
            fmt=" ",
            lw=1,
            alpha=0.5,
            color="k",
            capsize=1,
        )
        ax.set_title(channel_pair)
        ax.set_xscale("log", base=2)


def plot_label_and_distance_feature_distributions_per_cycle(df):
    df_object = df.select(
        sel.hierarchy_index | META_COLUMNS | sel.label | sel.distance
    ).pipe(unnest_all_structs)

    g = sns.FacetGrid(
        df_object.melt(
            id_vars=df_object.lazy().select(sel.index | META_COLUMNS).columns,
            value_vars=df_object.lazy().select(sel.features).columns,
            variable_name="feature",
            value_name="value",
        ).to_pandas(),
        col="feature",
        col_wrap=4,
        hue="cycle",
        sharex=False,
        sharey=False,
        palette="plasma",
    )
    g.map(sns.kdeplot, "value", cut=0)


def plot_correlation_feature_distributions_per_cycle(df):
    df_corr = (
        df.select(sel.hierarchy_index | META_COLUMNS | sel.correlation)
        .pipe(
            stack_column_name_to_column,
            column_name="channel_pair",
            index=sel.hierarchy_index | META_COLUMNS,
        )
        .pipe(split_channel_pair_column)
    )

    g = sns.FacetGrid(
        df_corr.melt(
            id_vars=df_corr.lazy().select(sel.index | META_COLUMNS).columns,
            value_vars=df_corr.lazy().select(sel.features).columns,
            variable_name="feature",
            value_name="value",
        ).to_pandas(),
        col="channel_pair",
        row="feature",
        col_order=["DAPI.0|DAPI.1", "DAPI.1|DAPI.2", "DAPI.1|DAPI.3"],
        hue="cycle",
        sharex=True,
        sharey=True,
        xlim=(-1.05, 1.05),
        palette="plasma",
    )
    g.map(sns.kdeplot, "value", cut=1, common_norm=False, cumulative=True, legend=True)


def plot_intensity_feature_distributions_per_cycle(df):
    # TODO: SHOULD NOT USE LOG-TRANSFORMED FEATURE
    df_intensity = (
        df.select(sel.hierarchy_index | META_COLUMNS | sel.intensity)
        .pipe(
            stack_column_name_to_column,
            column_name="channel",
            index=sel.hierarchy_index | META_COLUMNS,
        )
        .pipe(split_channel_column)
    )  # .idx.set_index(sel.index.object | cs.by_name('acquisition'))

    g = sns.FacetGrid(
        df_intensity.select(
            # TODO: AUTO CLIP
            sel.index | META_COLUMNS,
            cs.by_name("Mean").clip(10**0.5, 10**3.5),
        ).to_pandas(),
        col="channel",
        col_wrap=4,
        hue="cycle",
        sharex=False,
        sharey=False,
        palette="plasma",
    )
    g.map(
        sns.kdeplot,
        "Mean",
        common_norm=False,
        cumulative=False,
        legend=True,
        log_scale=True,
        cut=0,
    )
