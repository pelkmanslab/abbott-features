"""Tests for valid manifest."""

import json
from pathlib import Path

import requests
from jsonschema import validate

import abbott_features

PACKAGE_DIR = Path(abbott_features.__file__).parent
MANIFEST_FILE = PACKAGE_DIR / "__FRACTAL_MANIFEST__.json"
with MANIFEST_FILE.open("r") as f:
    MANIFEST = json.load(f)


def test_valid_manifest(tmp_path):
    """Test if the manifest is valid according to the JSON schema.

    NOTE: to avoid adding a fractal-server dependency, we simply download the
    relevant file.
    """
    # Download JSON Schema for ManifestV2
    url = (
        "https://raw.githubusercontent.com/fractal-analytics-platform/"
        "fractal-server/main/"
        "fractal_server/json_schemas/manifest_v2.json"
    )
    r = requests.get(url)
    with (tmp_path / "manifest_schema.json").open("wb") as f:
        f.write(r.content)
    with (tmp_path / "manifest_schema.json").open("r") as f:
        manifest_schema = json.load(f)

    validate(instance=MANIFEST, schema=manifest_schema)
