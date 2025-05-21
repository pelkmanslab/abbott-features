import shutil
from pathlib import Path

import pytest
from devtools import debug
from ngio.tables.tables_container import open_table
from ngio.utils._errors import NgioValueError
from pydantic import ValidationError

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
    zarr_urls = [f"{test_data_dir}/B/03/0", f"{test_data_dir}/B/03/1"]

    measure_intensity_features = IntensityFeaturesInputModel(
        channels_to_include=[
            ChannelInputModel(label="DAPI_2"),
            ChannelInputModel(label="pSmad15_3"),
        ]
    )
    measure_distance_features = DistanceFeaturesInputModel(label_name_to="emb_linked")
    measure_colocalization_features = ColocalizationFeaturesInputModel(
        channel_pair=[
            ChannelPairInputModel(
                channel0=ChannelInputModel(label="DAPI_2"),
                channel1=ChannelInputModel(label="DAPI_3"),
            )
        ]
    )
    measure_neighborhood_features = NeighborhoodFeaturesInputModel(
        measure=False, label_img_mask=None
    )

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
            measure_distance_features=measure_distance_features,
            measure_colocalization_features=measure_colocalization_features,
            measure_neighborhood_features=measure_neighborhood_features,
            overwrite=True,
        )

    # Test exclude channels for intensity features
    measure_intensity_features = IntensityFeaturesInputModel(
        channels_to_exclude=[
            ChannelInputModel(label="DAPI_2"),
        ]
    )

    measure_features(
        zarr_url=zarr_urls[0],
        label_name="nuclei",
        reference_acquisition="0",
        level="0",
        use_masks=True,
        masking_label_name="emb_linked",
        ROI_table_name="emb_ROI_table_2_linked",
        measure_label_features=False,
        measure_intensity_features=measure_intensity_features,
        overwrite=True,
    )
    store = Path(zarr_urls[0]) / "tables/nuclei"
    table_loaded = open_table(store=store)
    assert table_loaded.dataframe.shape == (5460, 17)

    # Test validation of measure_neighborhood_features if measure is False
    with pytest.raises(ValidationError):
        NeighborhoodFeaturesInputModel(measure=False, label_img_mask="emb_linked")

    # Test overwrite set to False
    with pytest.raises(NgioValueError):
        measure_features(
            zarr_url=zarr_urls[0],
            label_name="nuclei",
            reference_acquisition="0",
            level="0",
            use_masks=True,
            masking_label_name="emb_linked",
            ROI_table_name="emb_ROI_table_2_linked",
            measure_label_features=True,
            overwrite=False,
        )

    # Test passing non masking_roi_table with use_masks=True
    with pytest.raises(NgioValueError):
        measure_features(
            zarr_url=zarr_urls[0],
            label_name="nuclei",
            reference_acquisition="0",
            level="0",
            use_masks=True,
            masking_label_name="emb_linked",
            ROI_table_name="FOV_ROI_table",
            measure_label_features=True,
            overwrite=False,
        )

    # Measure embryo level features
    measure_features(
        zarr_url=zarr_urls[0],
        label_name="emb_linked",
        reference_acquisition="0",
        level="0",
        use_masks=True,
        masking_label_name="emb_linked",
        ROI_table_name="emb_ROI_table_2_linked",
        measure_label_features=True,
        overwrite=True,
    )
