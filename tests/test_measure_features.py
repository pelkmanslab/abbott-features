import shutil
from pathlib import Path

import pytest
from devtools import debug

from abbott_features.fractal_tasks.measure_label_features import measure_label_features


@pytest.fixture(scope="function")
def test_data_dir(tmp_path: Path, zenodo_zarr: list) -> str:
    """
    Copy a test-data folder into a temporary folder.
    """
    zenodo_zarr_url = zenodo_zarr[0]
    dest_dir = (tmp_path / "data").as_posix()
    debug(zenodo_zarr_url, dest_dir)
    shutil.copytree(zenodo_zarr_url, dest_dir)
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
