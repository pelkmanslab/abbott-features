import shutil
from pathlib import Path

import pytest
from devtools import debug
from ngio.tables.tables_container import open_table

from abbott_features.fractal_tasks.fractal_utils import (
    ChannelInputModel,
    ChannelPairInputModel,
)
from abbott_features.fractal_tasks.measure_colocalization_features import (
    measure_colocalization_features,
)
from abbott_features.fractal_tasks.measure_distance_features import (
    measure_distance_features,
)
from abbott_features.fractal_tasks.measure_intensity_features import (
    measure_intensity_features,
)
from abbott_features.fractal_tasks.measure_label_features import (
    measure_label_features,
)

# @pytest.fixture(scope="function")
# def test_data_dir(tmp_path: Path, zenodo_zarr: list) -> str:
#     """
#     Copy a test-data folder into a temporary folder.
#     """
#     zenodo_zarr_url = zenodo_zarr[0]
#     dest_dir = (tmp_path / "data").as_posix()
#     debug(zenodo_zarr_url, dest_dir)
#     shutil.copytree(zenodo_zarr_url, dest_dir)
#     return dest_dir


@pytest.fixture(scope="function")
def test_data_dir(tmp_path: Path, zenodo_zarr: Path) -> str:
    """
    Copy a test-data folder into a temporary folder.
    """
    dest_dir = (tmp_path / "data").as_posix()
    debug(zenodo_zarr, dest_dir)
    shutil.copytree(zenodo_zarr, dest_dir)
    return dest_dir


def test_measure_label_features(test_data_dir):
    # Task-specific arguments
    zarr_url = f"{test_data_dir}/B/03/0"

    measure_label_features(
        zarr_url=zarr_url,
        label_name="nuclei",
        level="0",
        ROI_table_name="FOV_ROI_table",
        output_table_name="label_features",
    )


def test_measure_intensity_features(test_data_dir):
    # Task-specific arguments
    zarr_url = f"{test_data_dir}/B/03/0"

    measure_intensity_features(
        zarr_url=zarr_url,
        label_name="nuclei",
        level="0",
        ROI_table_name="FOV_ROI_table",
        output_table_name="intensity_features",
    )
    table = open_table(store=f"{zarr_url}/tables/intensity_features")
    # Validate polars dataframe shape
    assert table.dataframe.shape == (5485, 33)


def test_measure_intensity_features_selected_channels(test_data_dir):
    # Task-specific arguments
    zarr_url = f"{test_data_dir}/B/03/0"
    channel1 = ChannelInputModel(
        label="DAPI_2",
    )
    channel2 = ChannelInputModel(
        label="ECadherin_2",
    )
    selected_channels = [channel1, channel2]

    # Test include selected channels
    measure_intensity_features(
        zarr_url=zarr_url,
        label_name="nuclei",
        channels_to_include=selected_channels,
        level="0",
        ROI_table_name="FOV_ROI_table",
        output_table_name="intensity_features",
    )
    table = open_table(store=f"{zarr_url}/tables/intensity_features")

    # Validate polars dataframe shape
    assert table.dataframe.shape == (5485, 33)

    # Test exclude selected channels
    selected_channels = [channel1]
    measure_intensity_features(
        zarr_url=zarr_url,
        label_name="nuclei",
        channels_to_exclude=selected_channels,
        level="0",
        ROI_table_name="FOV_ROI_table",
        output_table_name="intensity_features",
    )
    table = open_table(store=f"{zarr_url}/tables/intensity_features")

    # Validate polars dataframe shape
    assert table.dataframe.shape == (5485, 17)


def test_measure_colocalization_features(test_data_dir):
    # Task-specific arguments
    zarr_url = f"{test_data_dir}/B/03/0"

    channel_pairs = ChannelPairInputModel(
        channel0=ChannelInputModel(label="DAPI_2"),
        channel1=ChannelInputModel(label="ECadherin_2"),
    )

    measure_colocalization_features(
        zarr_url=zarr_url,
        label_name="nuclei",
        channel_pairs=[channel_pairs],
        level="0",
        ROI_table_name="FOV_ROI_table",
        output_table_name="colocalization_features",
    )


def test_measure_distance_features(test_data_dir):
    # Task-specific arguments
    zarr_url = f"{test_data_dir}/B/03/0"

    measure_distance_features(
        zarr_url=zarr_url,
        label_name="nuclei",
        label_name_to="emb_linked",
        level="0",
        ROI_table_name="emb_ROI_table_2_linked",
        output_table_name="colocalization_features",
    )
