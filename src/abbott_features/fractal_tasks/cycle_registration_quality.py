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
"""Calculates registration score as quality control for registration across cycles."""

import logging

import polars as pl
from ngio import open_ome_zarr_container
from ngio.tables.v1 import FeatureTableV1
from pydantic import validate_call

from abbott_features.features.colocalization import (
    get_colocalization_features,
)
from abbott_features.fractal_tasks.io_models import InitArgsRegistration

logger = logging.getLogger(__name__)


@validate_call
def cycle_registration_quality(
    *,
    # Fractal arguments
    zarr_url: str,
    init_args: InitArgsRegistration,
    # Core parameters
    level: int,
    wavelength_id: str,
    label_name: str = "embryo",
    roi_table: str = "embryo_linked_ROI_table",
    masking_label_name: str = "embryo_linked",
    output_table_name: str = "cycle_registration_qc_table",
    overwrite: bool = True,
) -> None:
    """Calculate registration score across cycles

    Returns a table with Correlation Scores (Pearson, Spearman, Kendall) for
    each ROI and cycle.

    Args:
        zarr_url: Path or url to the individual OME-Zarr image to be processed.
            (standard argument for Fractal tasks, managed by Fractal server).
        init_args: Intialization arguments provided by
            `image_based_registration_hcs_init`. They contain the
            reference_zarr_url that is used for registration.
            (standard argument for Fractal tasks, managed by Fractal server).
        level: Pyramid level of the image to be used for registration.
        wavelength_id: Wavelength that will be used for image-based
            registration; e.g. `A01_C01` for Yokogawa, `C01` for MD.
        label_name: Name of the label to be used for masking.
        roi_table: Name of the ROI table over which the task loops to
            calculate the registration. Examples: `FOV_ROI_table` => loop over
            the field of views, `well_ROI_table` => process the whole well as
            one image.
        masking_label_name: Label for masking ROI e.g. `embryo`.
        output_table_name: Name of the output table to be saved in the
            reference OME-Zarr container.ƒ
        overwrite: If `True`, overwrite existing registration files.
            Default: `False`.
    """
    logger.info(
        f"Running for {zarr_url=}.\n"
        f"Calculating registration score per {roi_table=} for "
        f"{wavelength_id=}."
    )

    reference_zarr_url = init_args.reference_zarr_url

    # Load channel to register by
    ome_zarr_ref = open_ome_zarr_container(reference_zarr_url)
    channel_index_ref = ome_zarr_ref.get_channel_idx(wavelength_id=wavelength_id)

    ome_zarr_mov = open_ome_zarr_container(zarr_url)
    channel_index_align = ome_zarr_mov.get_channel_idx(wavelength_id=wavelength_id)

    # Read ROIs
    ref_roi_table = ome_zarr_ref.get_masking_roi_table(roi_table)
    logger.info(
        f"Found {len(ref_roi_table.rois())} ROIs in {roi_table=} to be processed."
    )

    label_image = ome_zarr_ref.get_masked_label(
        label_name=label_name,
        masking_label_name=masking_label_name,
        masking_table_name=roi_table,
        path=str(level),
    )

    logger.info(f"Start of registration score estimation for {zarr_url=} ")

    colocalization_table = []
    num_ROIs = len(ref_roi_table.rois())
    for i_ROI, ref_roi in enumerate(ref_roi_table.rois()):
        logger.info(
            f"Now processing ROI {i_ROI+1}/{num_ROIs} " f"for {wavelength_id=}."
        )

        ##############
        #  Calculate the registration score
        ##############
        channel_ref_lbl = ome_zarr_ref.channel_labels[channel_index_ref]
        channel_mov_lbl = ome_zarr_mov.channel_labels[channel_index_align]

        channel_ref = {
            "channel_label": channel_ref_lbl,
            "channel_zarr_url": reference_zarr_url,
        }

        channel_mov = {
            "channel_label": channel_mov_lbl,
            "channel_zarr_url": zarr_url,
        }

        # For now disable decay correction, could be added in future
        kwargs_decay_corr = {
            "t_decay_correction_df": None,
            "z_decay_correction": None,
        }

        colocalization_roi_table = get_colocalization_features(
            label_image=label_image,
            masking_label_name=masking_label_name,
            masking_table_name=roi_table,
            channel0=channel_ref,
            channel1=channel_mov,
            level=str(level),
            roi=ref_roi,
            kwargs_decay_corr=kwargs_decay_corr,
        )
        colocalization_table.append(colocalization_roi_table)

    # Combine results across ROIs
    if colocalization_table:
        table_out = pl.concat(colocalization_table)

        # Save results
        feature_table = FeatureTableV1(table_out, reference_label=label_name)
        ome_zarr_mov.add_table(
            name=output_table_name,
            table=feature_table,
            backend="experimental_parquet_v1",
            overwrite=overwrite,
        )

        logging.info("Finished saving table")


if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(
        task_function=cycle_registration_quality,
        logger_name=logger.name,
    )
