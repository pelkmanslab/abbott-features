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
import os
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import polars as pl
import polars.selectors as cs
import seaborn as sns
from fractal_tasks_core.cellvoyager.metadata import parse_yokogawa_metadata
from ngio import open_ome_zarr_plate
from ngio.tables.v1 import GenericTable
from pydantic import validate_call

from abbott_features.fractal_tasks.io_models import AcquisitionFolderInputModel
from abbott_features.intensity_normalization.polars_selector import sel
from abbott_features.intensity_normalization.polars_utils import (
    get_correlation_by_acquisition_map,
    plot_channel_t_decay_models,
    split_channel_column,
    stack_column_name_to_column,
    stack_correlation_metric_by_acquisition,
    to_tall,
    unnest_structs,
)

logger = logging.getLogger(__name__)


@validate_call
def cellvoyager_time_decay(
    *,
    # Fractal parameters
    zarr_urls: list[str],
    zarr_dir: str,
    # Task-specific arguments:
    acquisition_params: list[AcquisitionFolderInputModel],
    mrf_filename: str = "MeasurementDetail.mrf",
    mlf_filename: str = "MeasurementData.mlf",
    feature_table_name: str = "nuclei",
    label_name: str = "nuclei",
    spherical_radius_cutoff: tuple[float, float] = (4, 8),
    control_wells: Optional[list[str]] = None,
    alignment_score_cutoff: float = 0.8,
    save_timepoints_table: bool = True,
    time_decay_table_name: str = "time_decay_models",
    overwrite: bool = True,
) -> None:
    """Measure features.

    This tasks loops over the ROIs in a given ROI table and measures colocalization
    features within the label image. The features are saved as a .parquet table
    to be used in combination with polars.

    Args:
        zarr_urls: List of paths or urls to the individual OME-Zarr image to
            be processed.
            (standard argument for Fractal tasks, managed by Fractal server.
            Not used in this task.).
        zarr_dir: path of the directory where the new OME-Zarrs will be
            created.
            (standard argument for Fractal tasks, managed by Fractal server).
        acquisition_params: A list of `AcquisitionFolderInputModel` s, taking
            the acquisition int and path to the folder that contains the Cellvoyager
            image files and the MeasurementData & MeasurementDetail metadata files.
        mrf_filename: Name of the .mrf file. Default is "MeasurementDetail.mrf".
        mlf_filename: Name of the .mlf file. Default is "MeasurementData.mlf".
        feature_table_name: Name of the feature table to be used for time decay
            calculations calculated in "Measure Features" task e.g. "nuclei".
        label_name: Name of the label image to be used for time decay calculations.
            Should match the label_name used in the "Measure Features" task.
        spherical_radius_cutoff: Tuple of floats defining the lower and upper
            cutoff for the equivalent spherical radius of the objects to be
            included in the time decay models (to exclude e.g. bright mitotic cells
            that could shift the embryo's mean intensity). Default is (4, 8).
        control_wells: List of wells to be excluded from the time decay
            models. If None, no wells are excluded. E.g. ["B03", "B04"].
        alignment_score_cutoff: Float value to filter out misaligned cells
            based on the alignment score. Default is 0.8.
        save_timepoints_table: Whether to save the acquisition timepoints table.
            Not needed by downstream tasks. Default is True.
        time_decay_table_name: Name of the output time decay models table.
        overwrite: Whether to overwrite an existing output time decay table.
    """
    logging.info("Starting cellvoyager_time_decay_init task")

    ome_zarr_plate = open_ome_zarr_plate(zarr_dir)
    tables_dir = Path(zarr_dir) / "tables"
    os.makedirs(tables_dir, exist_ok=True)

    logging.info("Start parsing metadata for acquisition timestamps")
    metadata_dfs = []
    for acquisition_param in acquisition_params:
        acquisition = acquisition_param.acquisition
        image_dir = Path(acquisition_param.image_dir)

        metadata, _ = parse_yokogawa_metadata(
            mrf_path=f"{image_dir}/{mrf_filename}",
            mlf_path=f"{image_dir}/{mlf_filename}",
        )

        metadata["acquisition"] = acquisition
        metadata_dfs.append(metadata)

    df_pd = pd.concat(metadata_dfs)

    logging.info("Finished parsing metadata for acquisition timestamps")

    # Convert to polars and rename columns
    df = pl.from_pandas(df_pd, include_index=True)
    df = df.select(
        [
            pl.col("well_id").alias("well"),
            pl.col("FieldIndex").alias("ROI"),
            pl.col("Time").alias("acquisitionTime"),
            pl.col("acquisition"),
        ]
    )

    # Measure timeDelta (in minutes) per acquisition
    df_timepoints = df.with_columns(
        [
            (
                pl.col("acquisitionTime")
                - pl.min("acquisitionTime").over("acquisition")
            ).alias("timeDelta"),
            (
                (
                    pl.col("acquisitionTime")
                    - pl.min("acquisitionTime").over("acquisition")
                ).dt.total_seconds()
                / 60
            )
            .cast(pl.Float32)
            .alias("timeDeltaMinutes"),
        ]
    )

    # Optionally save the timepoints table
    if save_timepoints_table is True:
        timepoints_table = GenericTable(df_timepoints)

        ome_zarr_plate.add_table(
            name="acquisition_times",
            table=timepoints_table,
            backend="experimental_parquet_v1",
            overwrite=overwrite,
        )

        logging.info(f"Saved timepoints table to {zarr_dir=}")

    logging.info(
        "Finished calculating acquisition times, starting to "
        f"calculate time decay models on {feature_table_name} table."
    )

    df_features_pd = ome_zarr_plate.concatenate_image_tables(
        table_name=feature_table_name,
        index_key="index",
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
        (
            pl.concat_str(
                [
                    pl.col("index"),
                    pl.col("ROI"),
                ],
                separator="_",
            ).alias("index")
        ),
        pl.col("ROI").cast(pl.Int16),
    )

    # Convert to tall format
    df = to_tall(df_features, index_key="index")

    # Select relevant features
    intensity_features = "^.*_Mean$"
    corr_features = "^.*_PearsonR$"

    features = [
        "Centroid",
        "PhysicalSize",
        "EquivalentSphericalRadius",
        "Roundness",
    ]

    df = (
        df.select(
            sel.index,
            *[pl.col(f) for f in features],
            cs.matches(intensity_features),
            cs.matches(corr_features),
        )
        .pipe(unnest_structs, cols=["Centroid"])
        .with_columns([pl.col("path_in_well").cast(pl.Int16)])
    ).rename({"path_in_well": "acquisition"})

    df_label = df.select(sel.index, sel.label)

    # Extract tables
    df_intensity = (
        df.select(sel.index - sel.hierarchy, sel.intensity)
        .pipe(stack_column_name_to_column, index=sel.index, column_name="channel")
        .pipe(split_channel_column)
    )

    CORRELATION_BY_ACQUISITION_MAP = get_correlation_by_acquisition_map(df)
    df_alignment = stack_correlation_metric_by_acquisition(
        df, corr_map=CORRELATION_BY_ACQUISITION_MAP
    )

    # Remove mitotic etc. labels
    ax = sns.kdeplot(data=df_label.to_pandas(), x="EquivalentSphericalRadius")
    y_lims = ax.get_ylim()
    plt.vlines(
        x=spherical_radius_cutoff,
        ymin=y_lims[0],
        ymax=y_lims[1],
        colors=["k"],
        linestyles=["dotted", "dashed"],
    )
    # Save plot
    output_plots_dir = Path(tables_dir) / "__plots"

    os.makedirs(output_plots_dir, exist_ok=True)
    plt.savefig(
        Path(output_plots_dir) / "equivalent_spherical_radius_cutoff.png",
        dpi=300,
    )
    plt.close()
    df_label_clean = df_label.filter(
        pl.col("EquivalentSphericalRadius").is_between(*spherical_radius_cutoff)
    )

    # Merge tables
    index = ["ROI", "object", "label", "well", "channel", "stain", "acquisition"]
    obj_index = ["ROI", "object", "label", "well"]

    df_clean = (
        df_intensity.with_columns(
            [
                pl.col("acquisition").cast(pl.UInt16),
            ]
        )
        .select(
            [
                pl.col(index),
                pl.col("Mean"),
            ]
        )
        .join(
            df_label_clean.select(
                [*obj_index, "Centroid-z", "PhysicalSize", "EquivalentSphericalRadius"]
            ),
            on=obj_index,
        )
        .join(df_alignment, on=obj_index)
        .join(
            df_timepoints.with_columns(
                [
                    pl.col("acquisition").cast(pl.UInt16),
                ]
            ).select(["well", "ROI", "acquisition", "timeDeltaMinutes"]),
            on=["well", "ROI", "acquisition"],
        )
    )

    # Remove control wells from time decay models if specified & misaligned cells
    if alignment_score_cutoff is not None:
        if control_wells is not None:
            logging.info(
                f"Excluding control wells {control_wells} from time decay models."
            )
            df_clean = df_clean.with_columns(
                pl.col("well").is_in(control_wells).alias("is_control_well")
            )
            df_clean = df_clean.filter(
                pl.col("alignmentScore") > alignment_score_cutoff
            ).filter(~pl.col("is_control_well"))
    else:
        df_clean = df_clean.filter(pl.col("alignmentScore") > alignment_score_cutoff)

    # Aggregate intensity measurements per site
    logging.info("Calculating mean intensity per embryo and channel.")
    df_mean_per_embryo = (
        df_clean.group_by(["ROI", "channel", "acquisition", "well"])
        .agg([pl.col("Mean").mean(), pl.col("timeDeltaMinutes").mean()])
        .with_columns(
            [
                ((pl.col("Mean") - pl.col("Mean").mean()) / pl.col("Mean").std())
                .over("channel")
                .alias("Mean_ZScore"),
                ((pl.col("Mean") - pl.col("Mean").mean()) / pl.col("Mean").std())
                .over(["channel", "acquisition"])
                .alias("Mean_CycleZScore"),
                (pl.col("Mean") / pl.col("Mean").mean())
                .over(["channel", "acquisition"])
                .alias("Mean_CycleMeanNorm"),
            ]
        )
    )

    # Fit models to data
    models, fig = plot_channel_t_decay_models(
        df_mean_per_embryo,
        output_dir=output_plots_dir,
    )

    # Store correction factors for each model and channel in a table
    results = []

    all_model_names = set()
    for ch_models in models.values():
        all_model_names.update(ch_models.keys())

    for ch in models.keys():
        res = df_mean_per_embryo.filter(pl.col("channel") == ch)

        for mdl_name in models[ch]:
            mdl = models[ch][mdl_name]
            if mdl is not None:
                new_values = mdl._correction_factor(
                    res[["timeDeltaMinutes"]].to_numpy()
                )
                new_column = pl.Series(
                    name=f"correctionFactor-{mdl_name}", values=new_values
                )
                res = res.with_columns([new_column])
            else:
                logging.warning(f"Model {mdl_name} for channel {ch} is None")
                # Add a column of zeros or NaN to maintain consistent structure
                new_column = pl.Series(
                    name=f"correctionFactor-{mdl_name}",
                    values=[float("nan")] * len(res),
                )
                res = res.with_columns([new_column])

        # For models that don't exist for this channel, add NaN columns
        for mdl_name in all_model_names:
            if mdl_name not in models[ch]:
                new_column = pl.Series(
                    name=f"correctionFactor-{mdl_name}",
                    values=[float("nan")] * len(res),
                )
                res = res.with_columns([new_column])

        results.append(res)

    results = pl.concat(results)

    # Save table
    time_decay_table = GenericTable(results)
    ome_zarr_plate.add_table(
        name=time_decay_table_name,
        table=time_decay_table,
        backend="experimental_parquet_v1",
        overwrite=overwrite,
    )

    logging.info("Finished saving time decay models to.")


if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(task_function=cellvoyager_time_decay)
