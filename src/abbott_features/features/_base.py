import re
from collections.abc import Sequence

import itk
import numpy as np
import polars as pl

from abbott_features.features.types import LabelImage, SpatialImage


def get_si_features_df(
    lbl_img: LabelImage,
    int_img: SpatialImage | None = None,
    *,
    props: set[str] | None = None,
    named_features: bool = True,
    object_column: bool = False,
    struct_index: bool = False,
) -> pl.DataFrame:
    """Get features from a label image."""
    lbl_img_itk = itk.image_from_xarray(lbl_img)
    if int_img is not None:
        int_img_itk: SpatialImage | None = itk.image_from_xarray(int_img)
    else:
        int_img_itk: SpatialImage | None = int_img
    df = get_itk_features_df(lbl_img_itk, int_img_itk, props=props)
    if named_features:
        if int_img is not None:
            df = df.select(
                [
                    pl.col("label"),
                    pl.exclude("label").name.prefix(f"{int_img.name}_"),
                ]
            )
    if object_column:
        df = df.with_columns(
            [
                pl.lit(lbl_img.name).alias("object"),
            ]
        ).select(
            [
                pl.col(["object", "label"]),
                pl.exclude(["object", "label"]),
            ]
        )
        if struct_index:
            df = df.select(
                [
                    pl.struct(("object", "label")).alias("index"),
                    pl.exclude(("object", "label")),
                ]
            )
    return df


def get_itk_features_df(
    lbl_img: itk.Image,
    int_img: itk.Image = None,
    *,
    props: set[str] | None = None,
) -> pl.DataFrame:
    props = set() if props is None else set(props)
    costly_features = _calculate_costly_features(props)
    LabelMapType = itk.LabelMap[
        itk.StatisticsLabelObject[itk.UL, lbl_img.GetImageDimension()]
    ]
    if int_img:
        filt = itk.LabelImageToStatisticsLabelMapFilter[
            type(lbl_img), type(int_img), LabelMapType
        ].New()
        filt.SetInput1(lbl_img)
        filt.SetInput2(int_img)
        filt.SetComputeHistogram(costly_features["histogram"])
    else:
        filt = itk.LabelImageToShapeLabelMapFilter[type(lbl_img), LabelMapType].New()
        filt.SetInput(lbl_img)
        filt.SetComputeOrientedBoundingBox(costly_features["obbx"])
    filt.SetComputeFeretDiameter(costly_features["feret_diameter"])
    filt.SetComputePerimeter(costly_features["perimeter"])
    filt.Update()
    lbl_map = filt.GetOutput()
    return _get_df_from_feature_labelmap(lbl_map, props)


def _calculate_costly_features(props: set[str]) -> dict[str, bool]:
    kwargs = {}
    kwargs["feret_diameter"] = "FeretDiameter" in props
    kwargs["perimeter"] = "Perimeter" in props
    kwargs["obbx"] = "OrientedBoundingBox" in props
    kwargs["histogram"] = "Histogram" in props
    return kwargs


def _get_df_from_feature_labelmap(
    lbl_map: itk.LabelMap,  # Statistics- or ShapeLabelMap
    props: set[str],
    include_itk_transforms: bool = False,
) -> pl.DataFrame:
    props.add("Label")
    lbl_obj_sample = lbl_map.GetLabelObject(lbl_map.GetLabels()[0])
    get_pattern = re.compile("|".join([f"^Get{prop}" for prop in props]))
    get_props = list(filter(get_pattern.search, dir(lbl_obj_sample)))
    if not include_itk_transforms:
        get_props = list(filter(_is_not_itk_transform, get_props))
    df = pl.DataFrame()
    for get_prop in get_props:
        prop = get_prop[3:]
        data = [getattr(lbl_map[lbl], get_prop)() for lbl in lbl_map.GetLabels()]
        data = [convert_itk_dtypes(d) for d in data]
        df = df.with_columns(
            [
                pl.Series(prop, data),
            ]
        )
    return df.select([pl.col("Label").alias("label"), pl.exclude("Label")])


def _is_not_itk_transform(prop: str) -> bool:
    return not prop.endswith("Transform")


def convert_itk_dtypes(feature) -> dict[str, int] | dict[str, float] | float | int:
    if isinstance(feature, itk.Region):
        return _convert_itk_bbx(feature)
    elif isinstance(feature, (itk.Point, itk.Index)):  # noqa: UP038
        return _convert_itk_point_vector_index(feature)
    elif isinstance(feature, itk.Vector):
        return _convert_itk_point_vector_index(feature, dims=("a", "b", "c"))
    elif isinstance(feature, itk.Matrix):
        return _convert_itk_matrix(feature)
    elif isinstance(feature, itk.FixedArray):
        return _convert_itk_fixed_array(feature)
    else:
        return feature


def _convert_itk_bbx(
    bbx: itk.Region, dims: Sequence[str] = ("x", "y", "z")
) -> dict[str, int]:
    assert isinstance(bbx, itk.Region), "`bbx` must be of type itk.Region"
    columns = {}
    for idx, dim in zip(bbx.GetIndex(), dims, strict=False):
        columns[f"lower-{dim}"] = idx
    for idx, dim in zip(bbx.GetUpperIndex(), dims, strict=False):
        columns[f"upper-{dim}"] = idx + 1  # add one for python-style index [start, end)
    return columns


def _convert_itk_point_vector_index(
    point_like, dims: Sequence[str] = ("x", "y", "z")
) -> dict[str, int] | dict[str, float]:
    point = tuple(point_like)
    return dict(zip(dims, point, strict=False))


def _convert_itk_matrix(
    matrix: itk.Matrix,
    outer_dims: Sequence[str] = ("a", "b", "c"),
    inner_dims: Sequence[str] = ("x", "y", "z"),
) -> dict[str, float]:
    assert isinstance(matrix, itk.Matrix), "`matrix` must be of type `itk.Matrix`"
    columns = {}
    for m, dim in zip(np.array(matrix), outer_dims, strict=False):
        points = _convert_itk_point_vector_index(tuple(m), dims=inner_dims)
        for k in points:
            columns[f"{dim}-{k}"] = points[k]
    return columns


def _convert_itk_fixed_array(
    array: itk.FixedArray,
    outer_dims: Sequence[str] = ("a", "b", "c"),
    inner_dims: Sequence[str] = ("x", "y", "z"),
) -> dict[str, float]:
    assert isinstance(
        array, itk.FixedArray
    ), "`matrix` must be of type `itk.FixedArray`"
    columns = {}
    array_np = np.array([tuple(array.GetElement(i)) for i in range(array.Size())])
    for m, dim in zip(array_np, outer_dims, strict=False):
        points = _convert_itk_point_vector_index(tuple(m), dims=inner_dims)
        for k in points:
            columns[f"{dim}-{k}"] = points[k]
    return columns
