"""Copyright 2022 (C) Friedrich Miescher Institute for Biomedical Research and

University of Zurich

Original authors:
Ruth Hornbachner <ruth.hornbachner@uzh.ch>

This file is part of Fractal and was originally developed by eXact lab S.r.l.
<exact-lab.it> under contract with Liberali Lab from the Friedrich Miescher
Institute for Biomedical Research and Pelkmans Lab from the University of
Zurich.
"""

from pathlib import Path

import zarr
from ngio import open_ome_zarr_container


def get_zarrurl_from_image_label(well_url: Path, channel_label: str, level: str = "0"):
    """Get the zarr_url for a specific iamge channel from an OME-Zarr file.

    Args:
        well_url: Path to well of OME-Zarr file e.g. /path_to_zarr/B/03.
        channel_label: Label of the channel to get zarr_url for.
        level: Pyramid level of the OME_Zarr image. Default is "0".

    Returns:
        zarr_url for the specified channel_label.
    """
    well_group = zarr.open(well_url, mode="r")
    for image in well_group.attrs["well"]["images"]:
        zarr_url = well_url.joinpath(well_url, image["path"])
        ome_zarr_container = open_ome_zarr_container(zarr_url)
        channel_labels = ome_zarr_container.get_image(path=level).channel_labels

        if channel_label in channel_labels:
            return zarr_url

    raise ValueError(
        f"Channel label '{channel_label}' does not exist in well '{well_url}'."
    )


def get_well_from_zarrurl(zarr_url: str):
    """Get the well from a zarr_url.

    Args:
        zarr_url: Zarr URL of the image channel e.g. /path_to_zarr/B/03/0/.

    Returns:
        Well: e.g. "B03"
    """
    row = Path(zarr_url).parent.parent.name
    column = Path(zarr_url).parent.name
    return f"{row}{column}"
