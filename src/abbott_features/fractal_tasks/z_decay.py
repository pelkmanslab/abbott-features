# Copyright 2022 (C) Friedrich Miescher Institute for Biomedical Research and
# University of Zurich
#
# Original authors:
# Ruth Hornbachner <ruth.hornbachner@uzh.ch>
#
# This file is part of Fractal and was originally developed by eXact lab S.r.l.
# <exact-lab.it> under contract with Liberali Lab from the Friedrich Miescher
# Institute for Biomedical Research and Pelkmans Lab from the University of
# Zurich.

"""Task to get time decay correction models for CellVoyager Acquisitions."""

import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import polars.selectors as cs
import seaborn as sns
from ngio import open_ome_zarr_plate
from pydantic import validate_call
from scipy import stats

from abbott_features.intensity_normalization.models import (
    Exp,
    Linear,
    LogLinear,
    fit_model_to_df,
    write_models,
)
from abbott_features.intensity_normalization.polars_selector import sel
from abbott_features.intensity_normalization.polars_utils import (
    split_channel_column,
    split_channel_pair_column,
    stack_column_name_to_column,
    to_tall,
    unnest_structs,
)

logger = logging.getLogger(__name__)


@validate_call
def z_decay(
    *,
    # Fractal parameters
    zarr_urls: list[str],
    zarr_dir: str,
    # Task-specific arguments:
    feature_table_name: str = "nuclei",
    label_name: str = "nuclei",
    embryo_label_name: str = "embryo",
    spherical_radius_cutoff: tuple[float, float] = (4, 8),
    roundness_cutoff: float = 0.8,
    control_wells: Optional[list[str]] = None,
    alignment_score_cutoff: float = 0.8,
    loss: Literal["linear", "huber"] = "huber",
    overwrite: bool = True,
) -> None:
    """Fit z-decay models to each channel label of a feature table.

    This task requires ...

    Args:
        zarr_urls: List of paths or urls to the individual OME-Zarr image to
            be processed.
            (standard argument for Fractal tasks, managed by Fractal server.
            Not used in this task.).
        zarr_dir: path of the directory where the new OME-Zarrs will be
            created.
            (standard argument for Fractal tasks, managed by Fractal server).
        feature_table_name: Name of the feature table to be used for time decay
            calculations calculated in "Measure Features" task e.g. "nuclei".
        label_name: Name of the label image to be used for time decay calculations.
            Should match the label_name used in the "Measure Features" task.
        embryo_label_name: Optional name of the embryo label image to be used
            for z-decay two-step models ()
        spherical_radius_cutoff: Tuple of floats defining the lower and upper
            cutoff for the equivalent spherical radius of the objects to be
            included in the time decay models (to exclude e.g. bright mitotic cells
            that could shift the embryo's mean intensity). Default is (4, 8).
        roundness_cutoff: Float value defining the cutoff for the roundness of
            the objects to be included in the time decay models.
        control_wells: List of wells to be excluded from the z-decay
            models. If None, no wells are excluded. E.g. ["B03", "B04"].
        alignment_score_cutoff: Float value to filter out misaligned cells
            based on the alignment score. Default is 0.8.
        save_timepoints_table: Whether to save the acquisition timepoints table.
            Not needed by downstream tasks. Default is True.
        loss: Loss function to use for the model fitting.
            Can be "linear" or "huber".
        overwrite: Whether to overwrite an existing output time decay table.
    """
    logging.info("Starting z_decay task")

    out_fld_models = Path(zarr_dir) / "models"
    logging.info(f"Saving models to: {out_fld_models}")
    out_fld_plots = out_fld_models / "__plots/z_decay"
    logging.info(f"Saving plots to: {out_fld_plots}")
    out_fld_plots.mkdir(parents=True, exist_ok=True)

    ome_zarr_plate = open_ome_zarr_plate(zarr_dir)

    # Load the uncorrected feature table
    df_features_pd = ome_zarr_plate.concatenate_image_tables(
        table_name=feature_table_name, index_key="label"
    )

    df_features = pl.from_pandas(
        df_features_pd.dataframe, include_index=True
    ).with_columns(
        pl.lit(label_name).alias("object"),
        (
            pl.concat_str(
                [
                    pl.col("row"),
                    pl.col("column"),
                ],
                separator="",
            ).alias("well")
        ),
        pl.col("ROI").cast(pl.Int16),
    )

    # Convert the feature table to tall format.
    df_features = to_tall(df_features, index_key="label")

    # Discard outliers
    # Save plots of EquivalentSphericalRadius and Roundness distributions
    ax = sns.kdeplot(df_features.to_pandas(), x="EquivalentSphericalRadius")
    y_lims = ax.get_ylim()
    plt.vlines(
        x=spherical_radius_cutoff,
        ymin=y_lims[0],
        ymax=y_lims[1],
        colors=["k"],
        linestyles=["dotted", "dashed"],
    )
    plt.savefig(
        Path(out_fld_plots) / "equivalent_spherical_radius_cutoff.png",
        dpi=300,
    )
    plt.close()

    ax = sns.kdeplot(df_features.to_pandas(), x="Roundness")
    y_lims = ax.get_ylim()
    plt.vlines(
        x=roundness_cutoff,
        ymin=y_lims[0],
        ymax=y_lims[1],
        colors=["k"],
        linestyles=["dotted", "dashed"],
    )
    plt.savefig(
        Path(out_fld_plots) / "roundness_cutoff.png",
        dpi=300,
    )
    plt.close()

    # Remove outliers based on spherical radius and roundness cutoffs.
    df_clean = df_features.filter(
        pl.col("EquivalentSphericalRadius").is_between(*spherical_radius_cutoff)
        & (pl.col("Roundness") >= roundness_cutoff)
    )
    logging.info(
        f"Removed {df_features.height - df_clean.height} outliers out of "
        f"{df_features.height} total objects."
    )

    # Remove control wells if specified.
    if control_wells is not None:
        logging.info(f"Excluding control wells {control_wells} from z decay models.")
        df_clean = df_clean.with_columns(
            pl.col("well").is_in(control_wells).alias("is_control_well")
        ).filter(~pl.col("is_control_well"))

    # Select relevant features and compute medium & embryo path length.
    features = [
        "Centroid",
        "PhysicalSize",
        "EquivalentSphericalRadius",
        "Roundness",
        f"{embryo_label_name}_CentroidDistanceToBorder",
        f"{embryo_label_name}_CentroidDistanceAlongZ",
    ]
    intensity_features = "^.*_Mean$"
    corr_features = "^.*_PearsonR$"

    embryo_path_name = "EmbryoPath"
    medium_path_name = "MediumPath"

    path_selector = cs.matches("EmbryoPath|MediumPath")

    df = (
        df_clean.select(
            sel.index,
            *[pl.col(f) for f in features],
            cs.matches(intensity_features),
            cs.matches(corr_features),
        )
        .pipe(unnest_structs, cols=["Centroid"])
        .with_columns(
            [
                (
                    pl.col("Centroid-z")
                    - pl.col(f"{embryo_label_name}_CentroidDistanceAlongZ")
                ).alias(medium_path_name),
                pl.col(f"{embryo_label_name}_CentroidDistanceAlongZ").alias(
                    embryo_path_name
                ),
            ]
        )
    )

    # Convert tables into tall (tidy) format.
    df_label = df.select(
        sel.index,
        sel.label,
        path_selector,
    )

    df_intensity = (
        df.select(sel.index - sel.hierarchy, sel.intensity)
        .pipe(stack_column_name_to_column, index=sel.index, column_name="channel")
        .pipe(split_channel_column)
    )
    df_corr = (
        df.select(sel.index - sel.hierarchy, sel.correlation)
        .pipe(stack_column_name_to_column, index=sel.index, column_name="channel_pair")
        .pipe(split_channel_pair_column)
    )

    df_distance = df.select(sel.index - sel.hierarchy, sel.distance).pipe(
        stack_column_name_to_column, index=sel.index, column_name="object_to"
    )

    # Join all tables
    df_merge = (
        df_label.join(
            df_intensity,
            on=["ROI", "object", "label", "well", "path_in_well"],
            how="left",
        )
        .join(
            df_corr, on=["ROI", "object", "label", "well", "path_in_well"], how="left"
        )
        .join(
            df_distance,
            on=["ROI", "object", "label", "well", "path_in_well"],
            how="left",
        )
        .with_columns(cs.by_name("PearsonR").fill_null(1.0))
        .select(sel.index, path_selector, sel.features)
    )

    # Filter out objects below the alignment score cutoff.
    df_merge_clean = df_merge.filter(pl.col("PearsonR") >= alignment_score_cutoff)
    logging.info(
        f"Removed {df_merge.height - df_merge_clean.height} objects "
        f"with alignment score < {alignment_score_cutoff} out of "
        f"{df_merge.height} total objects."
    )

    # Calculate models
    logging.info("Start calculating z decay models...")

    # Get all unique channel labels from the dataframe
    channels = df_merge_clean.select(pl.col("channel")).unique().to_series().to_list()

    models = [
        Linear(loss=loss),
        LogLinear(loss=loss),
        Exp(loss=loss, pos_offset=True),
        Exp(loss=loss, pos_offset=False),
    ]

    def get_X_grid(X, n_samples: int = 200):
        return np.meshgrid(*np.linspace(X.min(axis=0), X.max(axis=0), n_samples).T)

    model_colors = ["#264653", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51"]
    scatter_color = "#8e8e8e"
    scatter_color_kde = False
    n_samples_kde = (
        10000 if df_merge_clean.height > 10000 else df_merge_clean.height // 2
    )
    n_samples_kde = min(n_samples_kde, df_merge_clean.height)

    X_columnss = [["Centroid-z"], ["MediumPath", "EmbryoPath"]]

    n_featuress = [len(X_columns) for X_columns in X_columnss]
    n_samples_model = None
    n_samples_plot = (
        20000 if df_merge_clean.height > 20000 else df_merge_clean.height // 2
    )

    if n_samples_kde > n_samples_plot:
        logging.info("`n_samples_kde` must be <= `n_samples_plot`.")
        n_samples_kde = n_samples_plot

    n_cols = 4

    # Calculate n_rows, ensuring we have enough cells for all channels
    n_rows = (len(channels) + n_cols - 1) // n_cols  # Ceiling division

    figs = [
        plt.figure(
            layout="constrained",
            figsize=(3.75 * n_cols, 2.25 * n_rows * n_features),
            dpi=200,
        )
        for n_features in n_featuress
    ]

    axss = [
        fig.subplots(
            n_rows,
            n_cols,
            subplot_kw={"projection": "3d"} if n_features == 2 else {},
            squeeze=False,
        ).reshape(-1)  # Flatten the array properly
        for fig, n_features in zip(figs, n_featuress)
    ]

    all_models = defaultdict(dict)
    for j, (model, model_color) in enumerate(zip(models, model_colors)):
        logger.info(model)
        for i, channel in enumerate(channels):
            # Skip if we've run out of axes
            if i >= len(axss[0]):
                logging.info(f"Warning: Not enough subplot space for channel {channel}")
                continue

            logging.info(channel)

            df_channel = (
                df_merge_clean.filter(pl.col("channel") == channel)
                .with_columns(pl.col("Mean").name.prefix(f"{channel}_"))
                .to_pandas()
            )

            y_column = f"{channel}_Mean"

            if n_samples_plot is not None:
                n_samples_plot = min(n_samples_plot, df_channel.shape[0])

            if n_samples_plot is None:
                df_plot = df_channel
            else:
                df_plot = df_channel.sample(n_samples_plot, random_state=42)

            for _, (X_columns, axs) in enumerate(zip(X_columnss, axss)):
                model_fit = fit_model_to_df(
                    df=df_channel,
                    channel=channel,
                    model=model,
                    X_columns=X_columns,
                    y_column=y_column,
                    n_samples=n_samples_model,
                    in_place=False,
                )
                name = str(model_fit)
                all_models[name][channel] = model_fit
                logging.info(name)

                X = df_plot[X_columns]
                y = df_plot[y_column]
                if scatter_color_kde and j == 0:
                    values = np.concatenate(
                        [X.values, np.expand_dims(y.values, axis=1)], axis=1
                    )
                    sample_idxs = np.random.choice(
                        np.arange(len(values)), n_samples_kde, replace=False
                    )
                    values_sample = values[sample_idxs, :]
                    kernel = stats.gaussian_kde(values_sample.T)
                    scatter_color = kernel(values.T)

                # Initialize y_top with a default value
                y_top = np.quantile(y, 0.999) * 1.5  # Default in case model_fit is None

                if model_fit is not None:
                    if len(X_columns) == 2:
                        X_grid_fit = get_X_grid(X)
                        y_grid_pred = model_fit.grid_predict(X_grid_fit)
                        y_top = np.quantile(y_grid_pred, 0.999) * 1.5
                    else:
                        X_fit = np.linspace(X.min(), X.max(), 200)
                        y_pred = model_fit.predict(X_fit)
                        y_top = np.quantile(y_pred, 0.999) * 1.5

                if len(X_columns) == 2:
                    ax = axs[i]
                    plt.sca(ax)
                    if j == 0:
                        ax.set_title(f"{channel}")
                        ax.scatter(
                            X.values[:, 0],
                            X.values[:, 1],
                            y,
                            c=scatter_color,
                            s=1,
                            alpha=0.3,
                        )

                    # Only attempt to plot the model surface if model_fit exists
                    if model_fit is not None:
                        label = "\n  ".join(
                            [
                                e
                                for e in re.split(r"[\(, \)]", str(model_fit))
                                if e != ""
                            ]
                        )
                        ax.plot_surface(
                            X_grid_fit[0],
                            X_grid_fit[1],
                            y_grid_pred,
                            color=model_color,
                            alpha=0.5,
                            label=label,
                        )
                    ax.set_zlim(0, y_top)

                    # Add legend for 3D plot (special handling required)
                    if j == len(models) - 1:  # Add legend on last model iteration
                        # Create proxy artists for legend
                        from matplotlib.lines import Line2D

                        legend_elements = []
                        for _, (mod, col) in enumerate(zip(models, model_colors)):
                            legend_elements.append(
                                Line2D(
                                    [0],
                                    [0],
                                    color=col,
                                    lw=2,
                                    label="\n".join(
                                        [
                                            e
                                            for e in re.split(r"[\(, \)]", str(mod))
                                            if e != ""
                                        ]
                                    ),
                                )
                            )
                        ax.legend(
                            handles=legend_elements, loc="upper right", fontsize="small"
                        )

                else:
                    ax = axs[i]
                    plt.sca(ax)
                    if j == 0:
                        ax.set_title(f"{channel}")
                        sns.scatterplot(
                            x=X.values.flatten(),
                            y=y,
                            c=scatter_color,
                            s=1,
                            alpha=0.3,
                        )
                    if model_fit is not None:
                        label = "\n  ".join(
                            [
                                e
                                for e in re.split(r"[\(, \)]", str(model_fit))
                                if e != ""
                            ]
                        )
                        plt.plot(
                            X_fit.flatten(),
                            y_pred,
                            color=model_color,
                            linewidth=1.5,
                            label=label,
                        )
                    plt.ylim(0, y_top)

                    # Add legend for 2D plot (only on the last model iteration)
                    if j == len(models) - 1:
                        ax.legend(fontsize="x-small")

            logging.info("")

    more_models = {}
    more_models["z_decay"] = all_models

    figs[0].savefig(out_fld_plots / "overview__one_step.png")
    figs[1].savefig(out_fld_plots / "overview__two_step.png")
    plt.close()
    logging.info(f"Saving models to {out_fld_models}")
    write_models(more_models, root=out_fld_models, overwrite=overwrite)

    logging.info("Finished z_decay task")


if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(task_function=z_decay)
