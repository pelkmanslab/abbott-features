import shutil
from pathlib import Path

import pytest
from devtools import debug
from ngio.tables.tables_container import open_table
from ngio.utils._errors import NgioValueError
from pydantic import ValidationError

from abbott_features.fractal_tasks.cellvoyager_time_decay import cellvoyager_time_decay
from abbott_features.fractal_tasks.io_models import (
    AcquisitionFolderInputModel,
    ChannelInputModel,
    ChannelPairInputModel,
    ColocalizationFeaturesInputModel,
    DistanceFeaturesInputModel,
    IntensityFeaturesInputModel,
    NeighborhoodFeaturesInputModel,
    TimeDecayInputModel,
)
from abbott_features.fractal_tasks.measure_features import measure_features
from abbott_features.fractal_tasks.z_decay import z_decay


@pytest.fixture(scope="function")
def test_data_dir(tmp_path: Path, zenodo_zarr: Path) -> str:
    """
    Copy a test-data folder into a temporary folder.
    """
    dest_dir = (tmp_path / "data").as_posix()
    debug(zenodo_zarr, dest_dir)
    shutil.copytree(zenodo_zarr, dest_dir)
    return dest_dir


def test_decays(test_data_dir):
    level = "0"
    zarr_urls = [f"{test_data_dir}/B/03/0", f"{test_data_dir}/B/03/1"]

    # First measure features
    measure_intensity_features = IntensityFeaturesInputModel(
        measure=True,
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
            level=level,
            use_masks=True,
            masking_label_name="emb_linked",
            ROI_table_name="emb_ROI_table_2_linked",
            measure_label_features=True,
            measure_intensity_features=measure_intensity_features,
            measure_colocalization_features=measure_colocalization_features,
            measure_distance_features=measure_distance_features,
            overwrite=True,
        )

    # Measure z decay models
    z_decay(
        zarr_urls=zarr_urls,
        zarr_dir=test_data_dir,
        feature_table_name="nuclei",
        label_name="nuclei",
        embryo_label_name="emb_linked",
        spherical_radius_cutoff=(2, 8),
        roundness_cutoff=0.8,
        alignment_score_cutoff=0,
        loss="huber",
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
        spherical_radius_cutoff=(2, 8),
    )

    # Measure features with z&t decay correction
    measure_intensity_features = IntensityFeaturesInputModel(
        measure=True,
        channels_to_include=[
            ChannelInputModel(label="DAPI_2"),
            ChannelInputModel(label="pSmad15_3"),
        ],
    )
    measure_colocalization_features = ColocalizationFeaturesInputModel(
        channel_pair=[
            ChannelPairInputModel(
                channel0=ChannelInputModel(label="DAPI_2"),
                channel1=ChannelInputModel(label="DAPI_3"),
            )
        ]
    )

    t_decay_correction = TimeDecayInputModel(
        correction_factor="correctionFactor-Linear", table_name="time_decay_models"
    )

    for zarr_url in zarr_urls:
        measure_features(
            zarr_url=zarr_url,
            label_name="nuclei",
            parent_label_names=["emb_linked"],
            reference_acquisition="0",
            level=level,
            use_masks=True,
            masking_label_name="emb_linked",
            ROI_table_name="emb_ROI_table_2_linked",
            measure_label_features=False,
            measure_intensity_features=measure_intensity_features,
            measure_colocalization_features=measure_colocalization_features,
            z_decay_correction="LogLinear(features=Centroid-z, loss=huber)",
            t_decay_correction=t_decay_correction,
            output_table_name="nuclei_t_corrected",
            overwrite=True,
        )


def test_measure_features(test_data_dir):
    level = "0"
    zarr_urls = [f"{test_data_dir}/B/03/0", f"{test_data_dir}/B/03/1"]

    measure_intensity_features = IntensityFeaturesInputModel(
        measure=True,
        channels_to_include=[
            ChannelInputModel(label="DAPI_2"),
            ChannelInputModel(label="pSmad15_3"),
        ],
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
        measure=True, label_img_mask="emb_linked"
    )

    for zarr_url in zarr_urls:
        measure_features(
            zarr_url=zarr_url,
            label_name="nuclei",
            parent_label_names=["emb_linked"],
            reference_acquisition="0",
            level=level,
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
        measure=True,
        channels_to_exclude=[
            ChannelInputModel(label="DAPI_2"),
        ],
    )

    measure_features(
        zarr_url=zarr_urls[0],
        label_name="nuclei",
        reference_acquisition="0",
        level=level,
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

    measure_intensity_features = IntensityFeaturesInputModel(
        measure=False,
    )

    # Test overwrite set to False
    with pytest.raises(NgioValueError):
        measure_features(
            zarr_url=zarr_urls[0],
            label_name="nuclei",
            reference_acquisition="0",
            level=level,
            use_masks=True,
            masking_label_name="emb_linked",
            ROI_table_name="emb_ROI_table_2_linked",
            measure_label_features=True,
            measure_intensity_features=measure_intensity_features,
            overwrite=False,
        )

    # Test passing non masking_roi_table with use_masks=True
    with pytest.raises(NgioValueError):
        measure_features(
            zarr_url=zarr_urls[0],
            label_name="nuclei",
            reference_acquisition="0",
            level=level,
            use_masks=True,
            masking_label_name="emb_linked",
            ROI_table_name="FOV_ROI_table",
            measure_label_features=True,
            overwrite=False,
        )
