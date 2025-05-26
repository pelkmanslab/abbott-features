import shutil
from pathlib import Path

import pytest
from devtools import debug

from abbott_features.fractal_tasks.cellvoyager_time_decay import cellvoyager_time_decay
from abbott_features.fractal_tasks.fractal_utils import (
    AcquisitionFolderInputModel,
    ChannelInputModel,
    ChannelPairInputModel,
    ColocalizationFeaturesInputModel,
    DistanceFeaturesInputModel,
    IntensityFeaturesInputModel,
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


def test_time_decay(test_data_dir):
    zarr_urls = [f"{test_data_dir}/B/03/0", f"{test_data_dir}/B/03/1"]

    # First measure features
    measure_intensity_features = IntensityFeaturesInputModel(
        channels_to_include=[
            ChannelInputModel(label="DAPI_2"),
            ChannelInputModel(label="pSmad15_3"),
        ]
    )
    measure_colocalization_features = ColocalizationFeaturesInputModel(
        channel_pair=[
            ChannelPairInputModel(
                channel0=ChannelInputModel(label="DAPI_2"),
                channel1=ChannelInputModel(label="DAPI_3"),
            )
        ]
    )
    measure_distance_features = DistanceFeaturesInputModel(label_name_to="emb_linked")

    for zarr_url in zarr_urls:
        measure_features(
            zarr_url=zarr_url,
            label_name="nuclei",
            parent_label_names=["emb_linked"],
            reference_acquisition="0",
            level="0",
            use_masks=True,
            masking_label_name="emb_linked",
            ROI_table_name="emb_ROI_table_2_linked",
            measure_label_features=True,
            measure_intensity_features=measure_intensity_features,
            measure_colocalization_features=measure_colocalization_features,
            measure_distance_features=measure_distance_features,
            overwrite=True,
        )

    # Next time decay
    acquisition_params = [
        AcquisitionFolderInputModel(
            acquisition=2, image_dir=str(Path(__file__).parent / "data/time_decay/c2/")
        ),
        AcquisitionFolderInputModel(
            acquisition=3, image_dir=str(Path(__file__).parent / "data/time_decay/c3/")
        ),
    ]

    cellvoyager_time_decay(
        zarr_urls=zarr_urls,
        zarr_dir=test_data_dir,
        acquisition_params=acquisition_params,
    )
