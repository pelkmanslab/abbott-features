"""Functions to extract distance features."""

from functools import partial
from typing import NamedTuple, Union

import itk
import numpy as np
import polars as pl
import spatial_image as si
from ngio.common import Roi
from ngio.images import Label
from ngio.images._masked_image import MaskedLabel

from abbott_features.features._base import get_si_features_df
from abbott_features.features.constants import (
    DefaultDistanceFeature,
    DefaultDistanceFunction,
    DistanceFeature,
    DistanceFunction,
)
from abbott_features.features.types import (
    BinaryImage,
    DistanceTransform,
    LabelImage,
    SpatialImage,
)


def _distance_to_border(mask: BinaryImage, label_image_to: Label) -> DistanceTransform:
    lbl_dim = "l"
    dt = itk.signed_maurer_distance_map_image_filter(mask, inside_is_positive=True)
    dt.coords["c"] = mask[lbl_dim].item()
    dt.name = label_image_to.meta.name
    return dt


def _distance_along_axis(
    mask: BinaryImage, label_image_to: Label, axis: str = "z"
) -> DistanceTransform:
    pixel_size = label_image_to.pixel_size.as_dict()[axis]
    sum_along_axis: SpatialImage = mask.cumsum(axis) * pixel_size
    return sum_along_axis.rename({"l": "c"})


DISTANCE_FUNCTIONS = {
    "DistanceToBorder": _distance_to_border,
    "DistanceAlongZ": _distance_along_axis,
    "DistanceAlongY": partial(_distance_along_axis, axis="y"),
    "DistanceAlongX": partial(_distance_along_axis, axis="x"),
}


def _get_mask(lbl_img: LabelImage, lbl: int, lbl_dim: str = "l") -> BinaryImage:
    mask = (lbl_img == lbl).astype(lbl_img.dtype)
    label_name = lbl_img.name
    mask.coords[lbl_dim] = label_name
    return mask


class LabelObject(NamedTuple):
    """A named tuple to hold the label and the corresponding image."""

    label_image: str
    label: int


def get_distance_features(
    label_image: Union[Label, MaskedLabel],
    label_image_to: Union[Label, MaskedLabel],
    roi: Roi,
    distance_transforms: tuple[DistanceFunction, ...] = tuple(DefaultDistanceFunction),
    features: tuple[DistanceFeature, ...] = tuple(DefaultDistanceFeature),
    named_features: bool = True,
    object_column: bool = False,
    struct_index: bool = False,
) -> pl.DataFrame:
    """Get distance features for a given label and

    its parent label (e.g. embryo mask).
    """
    # Load meta data
    dims = label_image.axes
    scale = label_image.pixel_size.as_dict()

    # Convert the label images to spatial_images
    if isinstance(label_image, MaskedLabel):
        label_numpy = label_image.get_roi_masked_as_numpy(int(roi.name)).astype(
            np.uint16
        )
    else:
        label_numpy = label_image.get_roi(roi).astype(np.uint16)

    label_spatial_image = si.to_spatial_image(
        label_numpy,
        dims=dims,
        scale=scale,
        name=label_image.meta.name,
    )
    if isinstance(label_image_to, MaskedLabel):
        label_numpy_to = label_image_to.get_roi_masked_as_numpy(int(roi.name)).astype(
            np.uint16
        )
    else:
        label_numpy_to = label_image_to.get_roi_as_numpy(roi).astype(np.uint16)
    label_spatial_image_to = si.to_spatial_image(
        label_numpy_to,
        dims=dims,
        scale=scale,
        name=label_image_to.meta.name,
    )

    distance_transforms = {k: DISTANCE_FUNCTIONS[str(k)] for k in distance_transforms}
    if struct_index:
        index = "index"
    elif object_column:
        index = ["object", "label"]
    else:
        index = "label"

    # TODO: needs to be better fixed to work with multiple labels in the same ROI
    # (e.g. multiple embryos in the same ROI) if not MaskedLabel is used.
    if isinstance(label_image_to, MaskedLabel):
        mask = _get_mask(label_spatial_image_to, int(roi.name))
    else:
        # Get unique labels that are not 0
        lbls_unique = np.unique(label_numpy)
        lbls_unique = lbls_unique[lbls_unique != 0]
        if len(lbls_unique) > 1:
            raise ValueError(
                f"Multiple labels found in label_image_to for ROI {roi.name}. "
                "Please use a MaskedLabel or ensure that there is only one label "
                f" in the ROI. Found labels: {lbls_unique}"
            )
        else:
            lbl_id = lbls_unique[0]
        mask = _get_mask(label_spatial_image_to, lbl_id)

    dfs = []
    for name, distance_function in distance_transforms.items():
        try:
            dt = distance_function(mask, label_image_to)
        except ValueError as e:
            print(f"Can't compute {name}")
            print(f"{e}")
            continue

        df = get_si_features_df(
            label_spatial_image,
            dt,
            props=features,
            named_features=named_features,
            object_column=object_column,
            struct_index=struct_index,
        )
        df = _get_distance_at_centroid(df, dt)
        dfs.append(df.select([pl.col(index), pl.exclude(index).name.suffix(name)]))
    df_out = pl.concat(
        [dfs[0].select(index), *[df.drop(index) for df in dfs]], how="horizontal"
    )

    df_out = df_out.with_columns(pl.lit(roi.name).alias("ROI"))
    return df_out


def _get_distance_at_centroid(df: pl.DataFrame, distance_transform: DistanceTransform):
    return df.with_columns(
        [
            pl.col("^.*Centroid$").map_elements(
                lambda x: _lookup_physical_point(x, distance_transform)
            ),
        ]
    )


def _lookup_physical_point(point: pl.Series, image: DistanceTransform):
    return image.sel(method="nearest", **point).item()
