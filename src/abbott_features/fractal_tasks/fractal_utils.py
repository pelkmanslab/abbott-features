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
from typing import Optional

import numpy as np
import polars as pl
import zarr
from ngio import open_ome_zarr_container


def get_zarrurl_from_image_label(
    well_url: Path,
    channel_label: str,
    zarr_ending: Optional[str] = None,
    level: str = "0",
):
    """Get the zarr_url for a specific iamge channel from an OME-Zarr file.

    Args:
        well_url: Path to well of OME-Zarr file e.g. /path_to_zarr/B/03.
        zarr_ending: Optional ending of the OME-Zarr file. E.g. "registered".
        channel_label: Label of the channel to get zarr_url for.
        level: Pyramid level of the OME_Zarr image. Default is "0".

    Returns:
        zarr_url for the specified channel_label.
    """
    well_group = zarr.open(well_url, mode="r")
    for image in well_group.attrs["well"]["images"]:
        print(image)
        if zarr_ending is not None and not image["path"].endswith(zarr_ending):
            continue
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


def pad_to_same_shape(np_array_1, np_array_2, np_array_3=None):
    """Pad two or three numpy arrays to the same shape with zeros.

    Args:
        np_array_1 (np.array): First numpy array to pad.
        np_array_2 (np.array): Second numpy array to pad.
        np_array_3 (np.array, optional): Third numpy array to pad.

    Returns:
        np.array, np.array[, np.array]: Arrays padded to the same shape.
    """
    arrays = [np_array_1, np_array_2] + ([np_array_3] if np_array_3 is not None else [])

    max_shape = arrays[0].shape
    for arr in arrays[1:]:
        max_shape = np.maximum(max_shape, arr.shape)

    def pad_array(arr):
        padded = np.zeros(max_shape, dtype=arr.dtype)
        padded[: arr.shape[0], : arr.shape[1], : arr.shape[2]] = arr
        return padded

    padded_arrays = [pad_array(arr) for arr in arrays]

    return tuple(padded_arrays)


def ensure_uint16(
    label_array: np.ndarray,
) -> tuple[np.ndarray, dict[int, int] | None]:
    """Return *label_array* as uint16, relabeling only when necessary.

    If the array dtype is already uint8 or uint16 **and** the maximum value
    fits in uint16, the original array is returned unchanged and the second
    element of the tuple is ``None`` (no remapping needed).

    Otherwise the array is compacted via :func:`_relabel_to_uint16` and the
    reverse mapping is returned so callers can restore original IDs.

    Args:
        label_array: Integer-typed numpy array of label IDs.

    Returns:
        (array, new_to_old) where *new_to_old* is ``None`` when no relabeling
        was performed.
    """
    if label_array.dtype in (np.uint8, np.uint16):
        return label_array, None
    max_val = int(label_array.max()) if label_array.size > 0 else 0
    if max_val <= np.iinfo(np.uint16).max:
        return label_array.astype(np.uint16), None
    relabeled, new_to_old = _relabel_to_uint16(label_array)
    return relabeled, new_to_old


def _relabel_to_uint16(
    label_array: np.ndarray,
) -> tuple[np.ndarray, dict[int, int]]:
    """Relabel a label array so that all IDs fit in uint16.

    Background (0) is preserved as 0.  All other unique values are mapped to a
    compact sequence starting at 1.

    Args:
        label_array: An integer-typed numpy array with arbitrary label values.

    Returns:
        relabeled: A uint16 numpy array with compacted label IDs.
        new_to_old: Mapping from new (uint16) label ID → original label ID,
            enabling feature tables to be remapped back to original IDs.
    """
    unique_labels = np.unique(label_array)
    # Exclude background
    foreground = unique_labels[unique_labels != 0]

    if len(foreground) > np.iinfo(np.uint16).max:
        raise ValueError(
            f"ROI contains {len(foreground)} unique label IDs, which exceeds "
            f"the uint16 maximum of {np.iinfo(np.uint16).max}. "
            "Cannot relabel to uint16."
        )

    # Build old→new and new→old lookup tables
    # new IDs start at 1
    new_ids = np.arange(1, len(foreground) + 1, dtype=np.uint16)
    old_to_new = dict(zip(foreground.tolist(), new_ids.tolist()))
    new_to_old = dict(zip(new_ids.tolist(), foreground.tolist()))

    # Vectorised relabel via a lookup table (fast for dense and sparse arrays)
    max_old = int(foreground.max()) if len(foreground) > 0 else 0
    lut = np.zeros(max_old + 1, dtype=np.uint16)
    for old, new in old_to_new.items():
        lut[old] = new

    # Values beyond lut length → background (they don't exist, but be safe)
    clipped = np.clip(label_array, 0, max_old).astype(np.intp)
    relabeled = lut[clipped]
    # Restore true background for any values that were clipped incorrectly
    relabeled[label_array == 0] = 0

    return relabeled, new_to_old


def remap_label_ids(table: pl.DataFrame, new_to_old: dict[int, int]) -> pl.DataFrame:
    _LABEL_ID_COL = "label"
    if _LABEL_ID_COL not in table.columns:
        raise ValueError(
            f"Relabeling to original label IDs failed: "
            f"Expected column '{_LABEL_ID_COL}' not found in table."
        )

    original_col_order = table.columns
    label_col_idx = original_col_order.index(_LABEL_ID_COL)

    mapping_df = pl.DataFrame(
        {
            _LABEL_ID_COL: list(new_to_old.keys()),
            "__original_label__": list(new_to_old.values()),
        },
        schema={
            _LABEL_ID_COL: pl.UInt16,
            "__original_label__": pl.UInt64,
        },
    )

    result = (
        table.join(mapping_df, on=_LABEL_ID_COL, how="left")
        .drop(_LABEL_ID_COL)
        .rename({"__original_label__": _LABEL_ID_COL})
    )

    # Restore original column order — join moves label to the end
    cols = result.columns
    cols.remove(_LABEL_ID_COL)
    cols.insert(label_col_idx, _LABEL_ID_COL)
    return result.select(cols)
