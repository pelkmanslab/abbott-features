"""Functions to extract colocalization features."""

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import (
    Literal,
    NamedTuple,
    TypeAlias,
    Union,
)

import ngio
import numpy as np
import polars as pl
from scipy.stats import chi2_contingency, kendalltau, pearsonr, spearmanr
from skimage.measure import regionprops_table
from sklearn.metrics import mutual_info_score, normalized_mutual_info_score

from abbott_features.features.constants import (
    ColocalizationFeature,
    DefaultColocalizationFeature,
)


# TODO: Consistency of error cases between statistics
def _pearsonr(x: Sequence[float], y: Sequence[float]) -> float:
    statistic, _ = pearsonr(x, y)
    return statistic


def _spearmanr(x: Sequence[float], y: Sequence[float]) -> float:
    statistic, _ = spearmanr(x, y)
    return statistic


def _kendalltau(x: Sequence[float], y: Sequence[float]) -> float:
    statistic, _ = kendalltau(x, y)
    return statistic


def _chi2_contingency(x: Sequence[float], y: Sequence[float], bins: int = 10) -> float:
    c_xy = np.histogram2d(x, y, bins=bins)[0]
    return chi2_contingency(c_xy)[0]


def _mutual_info(x: Sequence[float], y: Sequence[float]) -> float:
    return mutual_info_score(x, y)


def _mutual_info_bins(x: Sequence[float], y: Sequence[float], bins: int = 10) -> float:
    c_xy = np.histogram2d(x, y, bins)[0]
    return mutual_info_score(None, None, contingency=c_xy)


def _normalized_mutual_info(
    x: Sequence[float], y: Sequence[float], bins: int = 10
) -> float:
    return normalized_mutual_info_score(x, y)


ColocalizationFn: TypeAlias = Callable[[Sequence[float], Sequence[float]], float]


ALL_CORRELATION_FUNCTIONS: dict[str, ColocalizationFn] = {
    ColocalizationFeature.PEARSON_R: _pearsonr,
    ColocalizationFeature.SPEARMAN_R: _spearmanr,
    ColocalizationFeature.KENDALL_TAU: _kendalltau,
    ColocalizationFeature.CHI2_CONTINGENCY: _chi2_contingency,
    ColocalizationFeature.MUTUAL_INFO: _mutual_info,
    ColocalizationFeature.MUTUAL_INFO_BINS: _mutual_info_bins,
    ColocalizationFeature.MUTUAL_INFO_NORMALIZED: _normalized_mutual_info,
}


class ChannelPair(NamedTuple):
    """A named tuple to hold the channel pair information."""

    channel0: str
    channel1: str


LABEL_IMAGE_COLUMN = "label_image"
LABEL_ID_COLUMN = "label"
RESOURCE_COLUMNS = ("channel0", "channel1")


def get_colocalization_features(
    label_image: Union[ngio.images.label.Label, ngio.images.masked_image.MaskedLabel],
    channel0: dict[str, Path],
    channel1: dict[str, Path],
    *,
    level: str,
    roi: ngio.common._roi.Roi,
    features: tuple[ColocalizationFeature, ...] = tuple(DefaultColocalizationFeature),
    index_columns: tuple[Literal["label", "label_image"], ...] = ("label",),
    index_prefix: str | None = None,
    add_resource_column: bool = False,
    resource_prefix: str | None = "resource",
    resource_in_name: bool = True,
    return_metadata: bool = False,
) -> pl.DataFrame:
    """Get colocalization features from a label image and two channels."""
    # Get the label image
    if isinstance(label_image, ngio.images.masked_image.MaskedLabel):
        lbls = label_image.get_roi_masked(int(roi.name)).astype("uint16")
    else:
        lbls = label_image.get_roi(roi).astype("uint16")

    # Get the channel images
    channel_0_images = ngio.open_ome_zarr_container(
        channel0["channel_zarr_url"]
    ).get_image(path=level)
    channel_0_idx = channel_0_images.meta.get_channel_idx(
        label=channel0["channel_label"]
    )
    img1 = (
        channel_0_images.get_roi(roi, c=channel_0_idx, mode="numpy")
        .astype("uint16")
        .squeeze()
    )

    channel_1_images = ngio.open_ome_zarr_container(
        channel1["channel_zarr_url"]
    ).get_image(path=level)
    channel_1_idx = channel_1_images.meta.get_channel_idx(
        label=channel1["channel_label"]
    )
    img2 = (
        channel_1_images.get_roi(roi, c=channel_1_idx, mode="numpy")
        .astype("uint16")
        .squeeze()
    )

    valid_features = tuple(ColocalizationFeature(e) for e in features)

    coloc_functions = {str(k): ALL_CORRELATION_FUNCTIONS[k] for k in valid_features}

    props = regionprops_table(lbls, properties=("label", "slice"))
    labels = props["label"]
    slices = props["slice"]

    df = pl.DataFrame()

    for metric, func in coloc_functions.items():
        corrs = []
        for slc, label in zip(slices, labels, strict=False):
            lbls_slc = lbls[slc]
            img1_slc = img1[slc]
            img2_slc = img2[slc]
            img1_px = img1_slc[np.where(lbls_slc == label)]
            img2_px = img2_slc[np.where(lbls_slc == label)]
            try:
                # Check for constant arrays which will cause correlation functions
                # to fail
                if np.all(img1_px == img1_px[0]) or np.all(img2_px == img2_px[0]):
                    # When one or both arrays are constant, correlation is undefined
                    res = np.nan
                else:
                    res = func(img1_px, img2_px)
            except ValueError:
                # This can happen when arrays have no variance or other
                # statistical requirements aren't met
                res = np.nan
            corrs.append(res)
        df = df.with_columns(pl.Series(metric, corrs))

    df = df.fill_nan(None)

    meta = {
        "feature_type": "correlation",
        "label_image": label_image.meta.name,
        "channel0": channel0["channel_label"],
        "channel1": channel1["channel_label"],
    }

    if resource_in_name:
        df = df.select(
            [
                pl.all().name.prefix(f"{meta['channel0']}|{meta['channel1']}_"),
            ]
        )

    if LABEL_ID_COLUMN in index_columns:
        df = df.with_columns(
            pl.Series(name=LABEL_ID_COLUMN, values=labels).cast(pl.Int64)
        )

    if LABEL_IMAGE_COLUMN in index_columns:
        df = df.with_columns(pl.lit(meta["label_image"]).alias(LABEL_IMAGE_COLUMN))

    if index_prefix is not None:
        df = df.select(
            pl.col(index_columns).name.prefix(f"{index_prefix}."),
            pl.exclude(index_columns),
        )
        index_columns_out = tuple(
            [f"{index_prefix}.{index_column}" for index_column in index_columns]
        )
    else:
        df = df.select(pl.col(index_columns), pl.exclude(index_columns))
        index_columns_out = index_columns

    if add_resource_column:
        df = df.with_columns(
            pl.lit(channel0.c.item()).alias(RESOURCE_COLUMNS[0]),
            pl.lit(channel1.c.item()).alias(RESOURCE_COLUMNS[1]),
        )

        if resource_prefix is not None:
            resource_columns_out = tuple(
                [
                    f"{resource_prefix}.{resource_column}"
                    for resource_column in RESOURCE_COLUMNS
                ]
            )
            df = df.select(
                pl.col(index_columns_out),
                pl.col(RESOURCE_COLUMNS).name.prefix(f"{resource_prefix}."),
                pl.exclude(index_columns_out + RESOURCE_COLUMNS),
            )
        else:
            resource_columns_out = RESOURCE_COLUMNS
    else:
        resource_columns_out = RESOURCE_COLUMNS

    meta["index_columns"] = index_columns_out
    meta["resource_columns"] = resource_columns_out

    # add ROI column
    df = df.with_columns(pl.lit(roi.name).alias("ROI"))

    if return_metadata:
        return df, meta
    return df
