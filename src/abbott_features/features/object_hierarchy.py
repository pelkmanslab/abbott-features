"""Function to extract object hierarchies across label images."""

from typing import Union

import numpy as np
import polars as pl
from ngio.common import Roi
from ngio.images import Label
from ngio.images._masked_image import MaskedLabel
from skimage.measure import regionprops

from abbott_features.fractal_tasks.fractal_utils import pad_to_same_shape


def parent_label(lbl: np.array, lbl2: np.array):
    labels, counts = np.unique(lbl2[np.where(lbl)], return_counts=True)
    parent_label = labels[np.argsort(counts)[-1]]
    return int(parent_label)


def get_parent_objects(
    label_image: Union[Label, MaskedLabel],
    parent_label_images: list[Union[Label, MaskedLabel]],
    roi: Roi,
    index_column: str = "label",
    parent_prefix: str | None = "parent",
) -> pl.DataFrame:
    if isinstance(label_image, MaskedLabel):
        label_numpy = label_image.get_roi_masked_as_numpy(int(roi.name))
    else:
        label_numpy = label_image.get_roi_as_numpy(roi)

    # results = {}
    # props = regionprops(label_numpy, cache=False)
    # labels = [prop.label for prop in props]
    # for lbl_other in parent_label_images:
    #     if isinstance(lbl_other, MaskedLabel):
    #         lbl_other_numpy = lbl_other.get_roi_masked_as_numpy(int(roi.name))
    #     else:
    #         lbl_other_numpy = lbl_other.get_roi_as_numpy(roi)

    #     # Pad to same shape if needed
    #     if label_numpy.shape != lbl_other_numpy.shape:
    #         label_numpy, lbl_other_numpy = pad_to_same_shape(label_numpy,
    #                                                          lbl_other_numpy)

    #     props = regionprops(
    #         label_numpy, lbl_other_numpy,
    #         cache=False, extra_properties=(parent_label,)
    #     )
    #     results[lbl_other.meta.name] = [prop.parent_label for prop in props]

    results = {}
    labels = None

    for lbl_other in parent_label_images:
        if isinstance(lbl_other, MaskedLabel):
            lbl_other_numpy = lbl_other.get_roi_masked_as_numpy(int(roi.name))
        else:
            lbl_other_numpy = lbl_other.get_roi_as_numpy(roi)

        # Pad in-place where possible to avoid holding two full copies
        if label_numpy.shape != lbl_other_numpy.shape:
            label_numpy, lbl_other_numpy = pad_to_same_shape(
                label_numpy, lbl_other_numpy
            )

        props = regionprops(
            label_numpy,
            lbl_other_numpy,
            cache=False,  # don't cache unused properties
            extra_properties=(parent_label,),
        )

        # Extract only what's needed immediately; don't hold regionprops object
        results[lbl_other.meta.name] = [prop.parent_label for prop in props]

        # Lazily grab labels from first iteration instead of a separate regionprops call
        if labels is None:
            labels = [prop.label for prop in props]  # already computed above

        del lbl_other_numpy  # release memory as soon as possible

    df = pl.DataFrame(results)

    if parent_prefix is not None:
        df = df.select(pl.all().name.prefix(f"{parent_prefix}."))

    # Set "label" as index column
    df = df.with_columns(pl.Series(name=index_column, values=labels).cast(pl.Int64))

    df = df.select(pl.col(index_column), pl.exclude(index_column))

    df = df.with_columns(pl.lit(roi.name).alias("ROI"))

    return df
