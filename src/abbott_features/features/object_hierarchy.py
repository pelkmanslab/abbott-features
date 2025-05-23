"""Function to extract object hierarchies across label images."""

from typing import Union

import ngio
import numpy as np
import polars as pl
from skimage.measure import regionprops


def parent_label(lbl: np.array, lbl2: np.array):
    labels, counts = np.unique(lbl2[np.where(lbl)], return_counts=True)
    parent_label = labels[np.argsort(counts)[-1]]
    return int(parent_label)


def get_parent_objects(
    label_image: Union[ngio.images.label.Label, ngio.images.masked_image.MaskedLabel],
    parent_label_images: list[
        Union[ngio.images.label.Label, ngio.images.masked_image.MaskedLabel]
    ],
    roi: ngio.common._roi.Roi,
    index_column: str = "label",
    parent_prefix: str | None = "parent",
) -> pl.DataFrame:
    if isinstance(label_image, ngio.images.masked_image.MaskedLabel):
        label_numpy = label_image.get_roi_masked(int(roi.name)).astype("uint16")
    else:
        label_numpy = label_image.get_roi(roi).astype("uint16")

    results = {}
    props = regionprops(label_numpy)
    labels = [prop.label for prop in props]
    for lbl_other in parent_label_images:
        if isinstance(lbl_other, ngio.images.masked_image.MaskedLabel):
            lbl_other_numpy = lbl_other.get_roi_masked(int(roi.name)).astype("uint16")
        else:
            lbl_other_numpy = lbl_other.get_roi(roi).astype("uint16")
        props = regionprops(
            label_numpy, lbl_other_numpy, extra_properties=(parent_label,)
        )
        results[lbl_other.meta.name] = [prop.parent_label for prop in props]

    df = pl.DataFrame(results)

    if parent_prefix is not None:
        df = df.select(pl.all().name.prefix(f"{parent_prefix}."))

    # Set "label" as index column
    df = df.with_columns(pl.Series(name=index_column, values=labels).cast(pl.Int64))

    df = df.select(pl.col(index_column), pl.exclude(index_column))

    df = df.with_columns(pl.lit(roi.name).alias("ROI"))

    return df
