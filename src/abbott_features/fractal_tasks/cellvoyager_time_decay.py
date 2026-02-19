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
from ngio import open_ome_zarr_plate
from ngio.hcs._plate import _build_extras, concatenate_image_tables_as
from ngio.tables.v1 import FeatureTableV1, GenericTable
from pydantic import validate_call

from abbott_features.fractal_tasks.cellvoyager_metadata import parse_yokogawa_metadata
from abbott_features.fractal_tasks.io_models import AcquisitionFolderInputModel
from abbott_features.intensity_normalization.polars_selector import sel
from abbott_features.intensity_normalization.polars_utils import (
    get_correlation_by_acquisition_map,
    plot_channel_t_decay_models,
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
    zarr_dir: str,  # Not used in this task
    # Task-specific arguments:
    reference_acquisition: int,
    acquisition_params: list[AcquisitionFolderInputModel],
    mrf_filename: str = "MeasurementDetail.mrf",
    mlf_filename: str = "MeasurementData.mlf",
    feature_table_name: str = "nuclei",
    label_name: str = "nuclei",
    spherical_radius_cutoff: tuple[float, float] = (4, 8),
    control_wells: Optional[list[str]] = None,
    alignment_score_cutoff: float = 0.8,
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
        reference_acquisition: The reference acquisition to be used for removing
            mitotic cells etc.
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
        time_decay_table_name: Name of the output time decay table.
        overwrite: Whether to overwrite an existing output time decay table.
    """
    logging.info("Starting cellvoyager_time_decay_init task")

    zarr_fld = Path(zarr_urls[0]).parent.parent.parent.as_posix()
    logging.info(f"Zarr folder: {zarr_fld}")

    # Check if zarr_url ends on digit or e.g. _registered
    zarr_ending = None
    zarr_stem = Path(zarr_urls[0]).stem
    if not zarr_stem[-1].isdigit():
        zarr_ending = zarr_stem.split("_", 1)[1]

    tables_dir = Path(zarr_fld) / "tables"  # Directory to save plots and tables
    output_plots_dir = Path(tables_dir) / "__plots"
    os.makedirs(output_plots_dir, exist_ok=True)
    logging.info(f"Saving plots to: {output_plots_dir}")

    # 1. Retrieve per-cycle acquisition times from Yokogawa metadata files
    logging.info("Start parsing metadata for acquisition timestamps")
    metadata_dfs = []
    for acquisition_param in acquisition_params:
        acq = acquisition_param.acquisition
        image_dir = Path(acquisition_param.image_dir)

        metadata, _ = parse_yokogawa_metadata(
            mrf_path=f"{image_dir}/{mrf_filename}",
            mlf_path=f"{image_dir}/{mlf_filename}",
        )

        metadata["acquisition"] = acq
        metadata_dfs.append(metadata)

    df_pd = pd.concat(metadata_dfs)

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

    # Save the timepoints table
    timepoints_table = GenericTable(df_timepoints)

    ome_zarr_plate = open_ome_zarr_plate(zarr_fld)
    ome_zarr_plate.add_table(
        name="acquisition_times",
        table=timepoints_table,
        backend="experimental_parquet_v1",
        overwrite=overwrite,
    )
    logging.info(
        "Finished parsing metadata for acquisition timestamps "
        f"Saved timepoints table to {zarr_fld=}"
    )

    # 2. Load (time-uncorrected) feature table per acquisition
    logging.info(
        f"Starting to calculate time decay models on {feature_table_name} table."
    )

    # Load reference acquisition features
    ref_images = ome_zarr_plate.get_images(reference_acquisition)
    if zarr_ending is not None:
        ref_images = {k: v for k, v in ref_images.items() if k.endswith(zarr_ending)}

    # Workaround if more than one path to image per acquisition exists
    df_features_pd_ref = concatenate_image_tables_as(
        images=ref_images.values(),
        extras=_build_extras(ref_images.keys()),
        table_cls=FeatureTableV1,
        name=feature_table_name,
        index_key="index",
    )

    # Convert to polars and rename/cast columns
    df_features_ref = pl.from_pandas(
        df_features_pd_ref.dataframe, include_index=True
    ).with_columns(
        pl.lit(label_name).alias("object"),
        pl.lit(reference_acquisition).alias("acquisition").cast(pl.UInt16),
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

    # Reference acquisition and path in well might not always match
    ref_acq_path_in_well = (
        df_features_ref.select("path_in_well").unique().to_series().to_list()[0]
    )
    if zarr_ending is not None:
        ref_acq_path, _ = ref_acq_path_in_well.split("_")
        ref_acq_path = int(ref_acq_path)
    else:
        ref_acq_path = int(ref_acq_path_in_well)

    # Loop over acquisitions, extract features and fit time decay models
    results_combined = []
    for acquisition_param in acquisition_params:
        acq = acquisition_param.acquisition
        logging.info(f"Start calculating time decay models for acquisition {acq}")
        if acq == reference_acquisition:
            df_features = df_features_ref
        else:
            acq_images = ome_zarr_plate.get_images(int(acq))
            # As above, workaround if more than one image path per acquisition exists
            if zarr_ending is not None:
                acq_images = {
                    k: v for k, v in acq_images.items() if k.endswith(zarr_ending)
                }

            df_features_acq_pd = concatenate_image_tables_as(
                images=acq_images.values(),
                extras=_build_extras(acq_images.keys()),
                table_cls=FeatureTableV1,
                name=feature_table_name,
                index_key="index",
            )

            df_features_acq = pl.from_pandas(
                df_features_acq_pd.dataframe, include_index=True
            ).with_columns(
                pl.lit(label_name).alias("object"),
                pl.lit(acq).alias("acquisition").cast(pl.UInt16),
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

            df_features = pl.concat(
                [df_features_ref, df_features_acq], how="diagonal_relaxed"
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
            .with_columns(
                [
                    (
                        pl.col("path_in_well").str.split("_").list.first()
                        if zarr_ending is not None
                        else pl.col("path_in_well")
                    )
                    .cast(pl.UInt16)
                    .alias(
                        "acquisition_path"
                    )  # To distinguish path from acquisition cycle
                ]
            )
        )

        # 3. Extract intensity and correlation features
        df_intensity = df.select(sel.index - sel.hierarchy, sel.intensity).pipe(
            stack_column_name_to_column, index=sel.index, column_name="channel"
        )

        CORRELATION_BY_ACQUISITION_MAP = get_correlation_by_acquisition_map(df)
        df_alignment = stack_correlation_metric_by_acquisition(
            df, corr_map=CORRELATION_BY_ACQUISITION_MAP
        )

        # Filter out mitotic cells based on equivalent spherical radius and save plot
        # Always use the reference acquisition to define the spherical radius cutoffs
        df_label = df.select(sel.index, sel.label)

        if acq != reference_acquisition:
            df_label_temp = df_label.filter(
                pl.col("acquisition") == reference_acquisition
            )
            labels_to_keep = (
                df_label_temp.filter(
                    pl.col("EquivalentSphericalRadius").is_between(
                        *spherical_radius_cutoff
                    )
                )
                .select("label")
                .to_series()
                .to_list()
            )
            df_label_clean = df_label.filter(pl.col("label").is_in(labels_to_keep))
        else:
            df_label_clean = df_label.filter(
                pl.col("EquivalentSphericalRadius").is_between(*spherical_radius_cutoff)
            )
            # Save plot
            _, ax = plt.subplots(figsize=(6, 4))
            ax = sns.kdeplot(data=df_label.to_pandas(), x="EquivalentSphericalRadius")
            y_lims = ax.get_ylim()
            plt.vlines(
                x=spherical_radius_cutoff,
                ymin=y_lims[0],
                ymax=y_lims[1],
                colors=["k"],
                linestyles=["dotted", "dashed"],
            )

            plt.savefig(
                output_plots_dir
                / f"{feature_table_name}_equivalent_spherical_radius_cutoff.png",
                dpi=300,
            )
            plt.close()

        # Merge cleaned label data with intensity and alignment data
        index = ["ROI", "object", "label", "well", "channel", "acquisition"]
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
                    [
                        *obj_index,
                        "Centroid-z",
                        "PhysicalSize",
                        "EquivalentSphericalRadius",
                    ]
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
            df_clean = df_clean.filter(
                pl.col("alignmentScore") > alignment_score_cutoff
            )

        # Drop reference channels if acquisition is not the reference acquisition
        if acq != reference_acquisition:
            df_clean = df_clean.filter(pl.col("acquisition") != reference_acquisition)

        logger.info(f"df_clean: {df_clean.select('acquisition').unique()}")
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
        ######
        # Start of time decay model fitting
        ######

        # Fit models to data
        models, _ = plot_channel_t_decay_models(
            df_mean_per_embryo,
            output_dir=output_plots_dir,
            time_decay_table_name=f"time_decay_models_c{acq}",
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
        results_combined.append(results)

    # Save table
    # Combine all results into a single DataFrame
    results_combined_pl = pl.concat(results_combined)
    time_decay_table = GenericTable(results_combined_pl)
    try:
        ome_zarr_plate.add_table(
            name=time_decay_table_name,
            table=time_decay_table,
            backend="experimental_parquet_v1",
            overwrite=overwrite,
        )
    except ValueError as e:
        logging.error(
            f"Error saving time decay table: {e}"
            "Make sure that the table does not already exist "
            "or set overwrite=True."
        )
        raise e

    logging.info("Finished saving time decay models to.")


if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(task_function=cellvoyager_time_decay)
