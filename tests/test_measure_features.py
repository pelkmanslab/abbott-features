import shutil
from pathlib import Path

import pytest
from devtools import debug

from abbott_features.fractal_tasks.fractal_utils import (
    ChannelInputModel,
    ChannelPairInputModel,
    ColocalizationFeaturesInputModel,
    DistanceFeaturesInputModel,
    IntensityFeaturesInputModel,
    NeighborhoodFeaturesInputModel,
)
from abbott_features.fractal_tasks.measure_features import measure_features


@pytest.fixture(scope="function")
def test_data_dir(tmp_path: Path, zenodo_zarr: Path) -> str:
    """
    Copy a test-data folder into a temporary folder.
    """
    dest_dir = (tmp_path / "data").as_posix()
    debug(zenodo_zarr, dest_dir)
    shutil.copytree(zenodo_zarr, dest_dir)
    return dest_dir


def test_measure_features(test_data_dir):
    # Task-specific arguments
    zarr_url = f"{test_data_dir}/B/03/0"

    measure_intensity_features = IntensityFeaturesInputModel(
        channels_to_include=[
            ChannelInputModel(label="DAPI_2"),
            ChannelInputModel(label="ECadherin_2"),
        ]
    )
    measure_distance_features = DistanceFeaturesInputModel(label_name_to="emb_linked")
    measure_colocalization_features = ColocalizationFeaturesInputModel(
        channel_pair=[
            ChannelPairInputModel(
                channel0=ChannelInputModel(label="DAPI_2"),
                channel1=ChannelInputModel(label="ECadherin_2"),
            )
        ]
    )
    measure_neighborhood_features = NeighborhoodFeaturesInputModel(
        measure=True, label_img_mask="emb_linked"
    )

    measure_features(
        zarr_url=zarr_url,
        label_name="nuclei",
        level="0",
        ROI_table_name="emb_ROI_table_2_linked",
        measure_label_features=True,
        measure_intensity_features=measure_intensity_features,
        measure_distance_features=measure_distance_features,
        measure_colocalization_features=measure_colocalization_features,
        measure_neighborhood_features=measure_neighborhood_features,
        overwrite=True,
    )
