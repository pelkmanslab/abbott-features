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

"""Task to measure features from 3D label images."""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl
from ngio import open_ome_zarr_container, open_ome_zarr_plate, open_ome_zarr_well
from ngio.tables.v1 import FeatureTableV1
from ngio.utils._errors import NgioValueError
from pydantic import validate_call

from abbott_features.features.colocalization import (
    get_colocalization_features,
)
from abbott_features.features.distance import (
    get_distance_features,
)
from abbott_features.features.intensity import (
    get_intensity_features,
)
from abbott_features.features.label import (
    get_label_features,
)
from abbott_features.features.neighborhood.neighborhood import (
    get_neighborhood_features,
)
from abbott_features.features.object_hierarchy import (
    get_parent_objects,
)
from abbott_features.fractal_tasks.fractal_utils import (
    get_well_from_zarrurl,
    get_zarrurl_from_image_label,
)
from abbott_features.fractal_tasks.io_models import (
    ColocalizationFeaturesInputModel,
    DistanceFeaturesInputModel,
    IntensityFeaturesInputModel,
    NeighborhoodFeaturesInputModel,
    TimeDecayInputModel,
)
from abbott_features.intensity_normalization.models import (
    read_models,
)

logger = logging.getLogger(__name__)


@validate_call
def measure_features(
    *,
    # Default arguments for fractal tasks:
    zarr_url: str,
    # Task-specific arguments:
    label_name: str,
    parent_label_names: Optional[list[str]] = None,
    reference_acquisition: Optional[int] = None,
    level: str = "0",
    measure_label_features: bool = False,
    measure_intensity_features: IntensityFeaturesInputModel = (
        IntensityFeaturesInputModel()
    ),
    measure_distance_features: Optional[DistanceFeaturesInputModel] = None,
    measure_colocalization_features: Optional[ColocalizationFeaturesInputModel] = None,
    measure_neighborhood_features: NeighborhoodFeaturesInputModel = (
        NeighborhoodFeaturesInputModel()
    ),
    z_decay_correction: Optional[str] = None,
    t_decay_correction: Optional[TimeDecayInputModel] = None,
    ROI_table_name: str,
    use_masks: bool = True,
    masking_label_name: Optional[str] = None,
    output_table_name: Optional[str] = None,
    overwrite: bool = True,
) -> None:
    """Measure features.

    This tasks loops over the ROIs in a given ROI table and measures colocalization
    features within the label image. The features are saved as a .parquet table
    to be used in combination with polars.

    Args:
        zarr_url: Absolute path to the OME-Zarr image.
        label_name: Label image to be measured.
        reference_acquisition: The reference acquisition that contains the label
            image and table to perform the measurement on. If not provided, the
            task assumes that each acquisition has its own label image and table.
        level: Level of the OME-Zarr label to copy from. Valid choices are
            "0", "1", etc. (depending on which levels are available in the
            OME-Zarr label).
        parent_label_names: List of parent label names relative to child `label_name`.
            If provided, the task will assign child labels to their parent labels
            based on overlap. This is useful for hierarchical label images
            e.g. embryo -> cells -> nuclei.
        measure_label_features: Whether to measure label features.
        measure_intensity_features: From which channels intensity features should be
            measured. If not provided, the task will not measure any intensity
            features.
        measure_distance_features: If `label_name_to` is provided, the task will measure
            distance features of `label_name` relative to `label_name_to`
            e.g. `embryo` or `organoid` segmentation.
        measure_colocalization_features: If `channel_pair` is set, the task will
            measure colocalization features per channel pair. E.g. colocalization
            between `channel_0` and `channel_1`.
        measure_neighborhood_features: If `measure` is set to True, neighborhood
            features will be measured. If neighborhood is measured in e.g. `embryo`
            or `organoid` segmentation provide the `label_img_mask`.
        z_decay_correction: Name of z-decay model to use. Models are stored in
            /path_to_zarr_plate/models/z_decay/ .
        t_decay_correction: Takes the time decay correction factor
            `correction_factor` and `table_name` of the dataframe that contains
            the correction factors.
        ROI_table_name: Name of the ROI table over which the task loops to
            measure label features. Examples: `FOV_ROI_table` => loop over
            the field of views, `organoid_ROI_table` => loop over the organoid
            ROI table (generated by another task), `well_ROI_table` => process
            the whole well as one image.
        use_masks: If `True`, try to use masked loading and fall back to
            `use_masks=False` if the ROI table is not suitable. Masked
            loading is relevant when only a subset of the bounding box should
            actually be processed (e.g. running within `organoid_ROI_table`).
        masking_label_name: Name of the masking label image to use for masked loading
            e.g. `embryo`.
        output_table_name: Name of the output feature table.
        overwrite: Whether to overwrite an existing output feature table.
    """
    logging.info("Starting measure_features task")
    logging.info(f"{zarr_url=}")

    zarr_plate = Path(zarr_url).parent.parent.parent
    ome_zarr_plate = open_ome_zarr_plate(zarr_plate)
    logging.info(f"{zarr_plate=}")

    well_url = Path(zarr_url).parent
    ome_zarr_well = open_ome_zarr_well(well_url)
    logging.info(f"{well_url=}")

    # Get ref_zarr_url of reference acquisition
    # where the label image and table is stored if provided
    # otherwise assumes each acquisition has its own label image and table
    if reference_acquisition is not None:
        ref_zarr_path = ome_zarr_well.paths(reference_acquisition)[0]
        if len(ref_zarr_path) > 1:
            raise ValueError(
                f"More than one path found for acquisition "
                f"{reference_acquisition=} in {well_url=}. "
            )
        elif len(ref_zarr_path) == 0:
            raise ValueError(
                f"No path found for acquisition "
                f"{reference_acquisition=} in {well_url=}. "
            )
        else:
            ref_zarr_path = ref_zarr_path[0]
            ref_zarr_url = (well_url / ref_zarr_path).as_posix()
    else:
        ref_zarr_url = zarr_url

    # Open the OME-Zarr container
    ome_zarr_container = open_ome_zarr_container(zarr_url)

    # Get OME-Zarr container of reference acquisition
    ome_zarr_container_ref = open_ome_zarr_container(ref_zarr_url)

    # Get ROI table to loop over and check if it is a masking ROI table if use_masks
    if use_masks:
        roi_table = ome_zarr_container_ref.get_masking_roi_table(ROI_table_name)

    else:
        roi_table = ome_zarr_container_ref.get_table(ROI_table_name)

    # Get the label image
    if use_masks:
        label_img = ome_zarr_container_ref.get_masked_label(
            label_name,
            masking_label_name=masking_label_name,
            masking_table_name=ROI_table_name,
            path=level,
        )
    else:
        label_img = ome_zarr_container_ref.get_label(label_name, path=level)

    # Check if the max label value exceeds uint16 range
    # Need to convert to uint16 as itk.LabelImageToShapeLabelMapFilter
    # does not support uint32
    label_da = label_img.get_array(mode="dask")
    uint16_max = int(np.iinfo(np.uint16).max)

    # If dtype already fits in uint16, skip any computation
    if not label_da.dtype == np.uint16 or not label_da.dtype == np.uint8:
        max_label_value = int(label_da.max().compute())
        if max_label_value > uint16_max:
            raise ValueError(
                f"Label image contains values ({max_label_value}) that exceed the "
                f"maximum allowed value ({uint16_max}) for processing with "
                "itk.LabelImageToShapeLabelMapFilter."
            )

    # Get the images
    images = ome_zarr_container.get_image(path=level)

    # Get channels to include/exclude
    if measure_intensity_features.measure:
        # If no channels to include or exclude, use all channels
        channel_labels = ome_zarr_container.get_image(path=level).channel_labels
        if measure_intensity_features.channels_to_include is not None:
            channel_labels_to_include = [
                c.label for c in measure_intensity_features.channels_to_include
            ]
            channel_labels = [
                c for c in channel_labels if c in channel_labels_to_include
            ]
        if measure_intensity_features.channels_to_exclude is not None:
            channel_labels_to_exclude = [
                c.label for c in measure_intensity_features.channels_to_exclude
            ]
            channel_labels = [
                c for c in channel_labels if c not in channel_labels_to_exclude
            ]

    # Initialize z-decay_correction and t-decay_correction if provided
    well = get_well_from_zarrurl(zarr_url)

    # Load time-decay correction dataframe if provided
    if t_decay_correction is not None:
        try:
            df_t_corr = (
                pl.from_pandas(
                    ome_zarr_plate.get_table(t_decay_correction.table_name).dataframe,
                    include_index=True,
                )
                .filter(pl.col("well") == well)
                .select(["ROI", "channel", t_decay_correction.correction_factor])
                .rename({t_decay_correction.correction_factor: "correctionFactor"})
            )

        except NgioValueError as err:
            raise KeyError(
                f"Time-decay correction table '{t_decay_correction}' not found in "
                f"the OME-Zarr plate at {zarr_plate}."
            ) from err

        logging.info(
            f"Using time-decay correction factors from {t_decay_correction=} table."
        )
    else:
        df_t_corr = None
        logging.info("No time-decay correction table provided, skipping.")

    # Load z-decay correction model if provided
    if z_decay_correction is not None:
        model_fld = Path(zarr_plate) / "models/z_decay" / z_decay_correction
        z_decay_model = read_models(model_fld)
        logging.info(f"{z_decay_model=}")
    else:
        z_decay_model = None
        logging.info("No z-decay correction model provided, skipping.")

    logging.info(f"Start feature measurement for {label_name=} and {zarr_url=}")

    num_ROIs = len(roi_table.rois())
    tables_list = []
    for i_ROI, roi in enumerate(roi_table.rois()):
        tables_roi_list = []
        logger.info(f"Now processing ROI {i_ROI+1}/{num_ROIs}")

        kwargs_decay_corr = {
            "t_decay_correction_df": df_t_corr,
            "z_decay_correction": z_decay_model,
        }

        # Measure features per ROI
        if parent_label_names is not None:
            if use_masks:
                parent_label_images = [
                    ome_zarr_container_ref.get_masked_label(
                        label_name,
                        masking_label_name=masking_label_name,
                        masking_table_name=ROI_table_name,
                        path=level,
                    )
                    for label_name in parent_label_names
                ]
            else:
                parent_label_images = [
                    ome_zarr_container_ref.get_label(label_name, path=level)
                    for label_name in parent_label_names
                ]

            hierarchy_roi_table = get_parent_objects(
                label_image=label_img,
                parent_label_images=parent_label_images,
                roi=roi,
            )
            tables_roi_list.append(hierarchy_roi_table)

        if measure_label_features:
            if zarr_url == ref_zarr_url:
                logging.info(f"Measure label features for {label_name=} in {zarr_url=}")
                label_roi_table = get_label_features(
                    label_image=label_img,
                    roi=roi,
                )
                tables_roi_list.append(label_roi_table)

        if measure_intensity_features.measure:
            if channel_labels:
                channel_roi_table_list = []
                for channel_label in channel_labels:
                    channel_roi_table = get_intensity_features(
                        label_image=label_img,
                        images=images,
                        channel_label=channel_label,
                        roi=roi,
                        kwargs_decay_corr=kwargs_decay_corr,
                    )

                    channel_roi_table_list.append(channel_roi_table)
                intensity_roi_table = pl.concat(channel_roi_table_list, how="align")
                tables_roi_list.append(intensity_roi_table)

        if measure_distance_features is not None:
            if zarr_url == ref_zarr_url:
                logging.info(
                    f"Measure distance features for {label_name=} in {zarr_url=}"
                )
                if use_masks:
                    label_img_to = ome_zarr_container_ref.get_masked_label(
                        measure_distance_features.label_name_to,
                        masking_label_name=masking_label_name,
                        masking_table_name=ROI_table_name,
                        path=level,
                    )
                else:
                    label_img_to = ome_zarr_container_ref.get_label(
                        measure_distance_features.label_name_to, path=level
                    )
                distance_roi_table = get_distance_features(
                    label_image=label_img, label_image_to=label_img_to, roi=roi
                )
                tables_roi_list.append(distance_roi_table)

        if measure_colocalization_features is not None:
            if zarr_url == ref_zarr_url:
                logging.info(
                    f"Measure colocalization features for {label_name=} in {zarr_url=}"
                )
                colocalization_roi_table_list = []
                for channel_pair in measure_colocalization_features.channel_pair:
                    channel_0_lbl = channel_pair.channel0.label
                    channel_1_lbl = channel_pair.channel1.label

                    channel_0_zarr_url = get_zarrurl_from_image_label(
                        well_url=well_url,
                        channel_label=channel_0_lbl,
                        level=level,
                    )
                    channel_0 = {
                        "channel_label": channel_0_lbl,
                        "channel_zarr_url": channel_0_zarr_url,
                    }

                    channel_1_zarr_url = get_zarrurl_from_image_label(
                        well_url=well_url,
                        channel_label=channel_1_lbl,
                        level=level,
                    )
                    channel_1 = {
                        "channel_label": channel_1_lbl,
                        "channel_zarr_url": channel_1_zarr_url,
                    }

                    colocalization_roi_table = get_colocalization_features(
                        label_image=label_img,
                        channel0=channel_0,
                        channel1=channel_1,
                        level=level,
                        roi=roi,
                        kwargs_decay_corr=kwargs_decay_corr,
                    )
                    colocalization_roi_table_list.append(colocalization_roi_table)

                colocalization_roi_table = pl.concat(
                    colocalization_roi_table_list, how="align"
                )
                tables_roi_list.append(colocalization_roi_table)

        if measure_neighborhood_features.measure:
            if zarr_url == ref_zarr_url:
                logging.info(
                    f"Measure neighborhood features for {label_name=} in {zarr_url=}"
                )
                label_img_mask = ome_zarr_container_ref.get_label(
                    measure_neighborhood_features.label_img_mask, path=level
                )
                neighborhood_table = get_neighborhood_features(
                    label_image=label_img,
                    label_img_mask=label_img_mask,
                    roi=roi,
                )
                tables_roi_list.append(neighborhood_table)

        if tables_roi_list:
            tables_roi = pl.concat(tables_roi_list, how="align")
            tables_list.append(tables_roi)

    logging.info(f"Finished feature measurement for {label_name=} and {zarr_url=}")

    if tables_list:
        table_out = pl.concat(tables_list)

        # Save the output table
        if output_table_name is None:
            output_table_name = label_name

        feature_table = FeatureTableV1(table_out, reference_label="label")
        ome_zarr_container.add_table(
            name=output_table_name,
            table=feature_table,
            backend="experimental_parquet_v1",
            overwrite=overwrite,
        )

        logging.info("Finished saving table")


if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(task_function=measure_features)
