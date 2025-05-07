import os
import shutil
from pathlib import Path

import pooch
import pytest


@pytest.fixture(scope="session")
def testdata_path() -> Path:
    TEST_DIR = Path(__file__).parent
    return TEST_DIR / "data/"


@pytest.fixture(scope="session")
def zenodo_zarr(testdata_path: Path) -> str:
    """
    This takes care of multiple steps:

    1. Download/unzip two Zarr containers (3D and MIP) from Zenodo, via pooch
    2. Copy the two Zarr containers into tests/data
    3. Modify the Zarrs in tests/data, to add whatever is not in Zenodo
    """

    # 1) Download Zarrs from Zenodo
    DOI = "10.5281/zenodo.15343366"
    DOI_slug = DOI.replace("/", "_").replace(".", "_")
    rootfolder = testdata_path / DOI_slug
    folder = rootfolder / "AssayPlate_Greiner_CELLSTAR655090.zarr"

    registry = {
        "AssayPlate_Greiner_CELLSTAR655090.zarr.zip": None,
    }
    base_url = f"doi:{DOI}"
    POOCH = pooch.create(
        pooch.os_cache("pooch") / DOI_slug,
        base_url,
        registry=registry,
        retry_if_failed=10,
        allow_updates=False,
    )

    file_name = "AssayPlate_Greiner_CELLSTAR655090.zarr"
    # 2) Download/unzip a single Zarr from Zenodo
    file_paths = POOCH.fetch(
        f"{file_name}.zip", processor=pooch.Unzip(extract_dir=file_name)
    )
    zarr_full_path = file_paths[0].split(file_name)[0] + file_name

    # 3) Copy the downloaded Zarr into tests/data
    if os.path.isdir(str(folder)):
        shutil.rmtree(str(folder))
    shutil.copytree(Path(zarr_full_path) / file_name, folder)
    return Path(folder)
