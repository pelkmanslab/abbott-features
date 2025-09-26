import shutil
from pathlib import Path

import pytest
from devtools import debug

from abbott_features.fractal_tasks.cycle_registration_quality import (
    cycle_registration_quality,
)
from abbott_features.fractal_tasks.init_registration_quality_hcs import (
    init_registration_quality_hcs,
)


@pytest.fixture(scope="function")
def test_data_dir(tmp_path: Path, zenodo_zarr: Path) -> str:
    """
    Copy a test-data folder into a temporary folder.
    """
    dest_dir = (tmp_path / "data.zarr").as_posix()
    debug(zenodo_zarr, dest_dir)
    shutil.copytree(zenodo_zarr, dest_dir)
    return dest_dir


def test_reg_quality(test_data_dir):
    level = 2
    reference_acquisition = 2
    zarr_urls = [f"{test_data_dir}/B/03/0", f"{test_data_dir}/B/03/1"]

    # Task-specific arguments
    wavelength_id = "A01_C01"
    label_name = "emb_linked"
    masking_label_name = "emb_linked"
    roi_table = "emb_ROI_table_2_linked"

    parallelization_list = init_registration_quality_hcs(
        zarr_urls=zarr_urls,
        zarr_dir="",
        reference_acquisition=reference_acquisition,
    )["parallelization_list"]

    for param in parallelization_list:
        cycle_registration_quality(
            zarr_url=param["zarr_url"],
            init_args=param["init_args"],
            level=level,
            wavelength_id=wavelength_id,
            label_name=label_name,
            masking_label_name=masking_label_name,
            roi_table=roi_table,
        )
