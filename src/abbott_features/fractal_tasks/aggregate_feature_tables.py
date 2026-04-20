"""This is the Python module for my_task."""

import logging
from pathlib import Path

import polars as pl
from ngio import OmeZarrContainer, open_ome_zarr_plate
from ngio.hcs._plate import _build_extras, concatenate_image_tables_as
from ngio.tables.v1 import FeatureTableV1
from pydantic import validate_call

from abbott_features.intensity_normalization.polars_utils import (
    to_tall,
)

logger = logging.getLogger("aggregate_feature_tables")


def _format_output_table_name(table_name_template: str, input_table_name: str) -> str:
    """Format the table name based on the provided template and input table name.

    Args:
        table_name_template (str): The template for the table name. This
        might contain a placeholder "{input_table_name}" which will be replaced
        by the input_table_name no placeholder at all,
        in which case the input_table_name will be ignored.
        input_table_name (str): The table name to insert into the
            table name template.

    Returns:
        str: The formatted output table name.
    """
    try:
        table_name = table_name_template.format(input_table_name=input_table_name)
    except KeyError as e:
        raise ValueError(
            "Table Name format error only allowed placeholder is "
            f"'input_table_name'. {{{e}}} was provided."
        ) from e
    return table_name


def concatenation_function(
    *,
    images: dict[str, OmeZarrContainer],
    input_table_name: str,
) -> pl.DataFrame:
    """Wrap concatenation of feature tables across OME-Zarr images into a single

    (tall) table.

    Args:
        images (str, OmeZarrContainer): Dictionary mapping image names to OME-Zarr
            Container containing the feature tables to concatenate.
        input_table_name (str): Name of the input feature table to concatenate across
            images.

    Returns:
        pl.DataFrame: Concatenated (tall) feature table across all images.
    """
    # Workaround if more than one path to image per acquisition exists
    df_features_pd = concatenate_image_tables_as(
        images=images.values(),
        extras=_build_extras(images.keys()),
        table_cls=FeatureTableV1,
        name=input_table_name,
        index_key="index",
        strict=False,
    )

    df_features = pl.from_pandas(
        df_features_pd.dataframe, include_index=True
    ).with_columns(
        pl.concat_str(["row", "column", "ROI", "label"], separator="_").alias("index")
    )

    df_features_tall = to_tall(df_features, index_key="index")

    return df_features_tall


@validate_call
def aggregate_feature_tables(
    *,
    # Fractal managed parameters
    zarr_urls: list[str],
    zarr_dir: str,  # Not used in this task
    # Aggregation parameters
    input_table_name: str,
    reference_label: str = "nuclei",
    output_table_name: str = "{input_table_name}_aggregated",
    overwrite: bool = True,
) -> None:
    """Aggregate feature tables across OME-Zarr images into a single (tall) table.

    Args:
        zarr_urls (list[str]): URLs to the OME-Zarr container
        zarr_dir: path of the directory where the new OME-Zarrs will be created.
            (standard argument for Fractal tasks, managed by Fractal server).
        input_table_name (str): Name of the input feature table to aggregate across
            images.
        reference_label (str): Name of the label the features in the input table
            refer to.
        output_table_name (str): Name of the output aggregated table.
            Defaults to "{input_table_name}_aggregated".
        overwrite (bool): Whether to overwrite an existing aggregated feature table.
    """
    logger.info("Starting to aggregate feature tables")

    zarr_fld = Path(zarr_urls[0]).parent.parent.parent.as_posix()
    logger.info(f"Zarr folder: {zarr_fld}")

    # Format the output table name based on the provided template
    output_table_name = _format_output_table_name(
        table_name_template=output_table_name, input_table_name=input_table_name
    )
    logger.info(f"Formatted output table name: {output_table_name=}")

    # Check if zarr_url ends on digit or e.g. _registered
    zarr_ending = None
    zarr_stem = Path(zarr_urls[0]).stem
    if not zarr_stem[-1].isdigit():
        if "_" in zarr_stem:
            zarr_ending = zarr_stem.split("_", 1)[1]

    ome_zarr_plate = open_ome_zarr_plate(zarr_fld)

    # Load reference acquisition features
    images = ome_zarr_plate.get_images()
    if zarr_ending is not None:
        images = {k: v for k, v in images.items() if k.endswith(zarr_ending)}

    concatenated_feature_table = concatenation_function(
        images=images,
        input_table_name=input_table_name,
    )

    concatenated_feature_table_out = FeatureTableV1(
        concatenated_feature_table,
        reference_label=reference_label,
    )

    ome_zarr_plate.add_table(
        name=output_table_name,
        table=concatenated_feature_table_out,
        backend="parquet",
        overwrite=overwrite,
    )

    logger.info(
        "Finished aggregating feature tables. "
        f"Saved aggregated table to {output_table_name}"
    )

    return None


if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(task_function=aggregate_feature_tables)
