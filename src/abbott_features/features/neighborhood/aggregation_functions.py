"""Aggregation functions for neighborhood features."""

import numba as nb
import numpy as np
from numpy.typing import NDArray

__all__ = [
    "Count",
    "Neighbors",
    "Min",
    "Median",
    "Max",
    "Quantile",
    "Mode",
    "Mean",
    "Std",
    "Var",
    "Sum",
    "CircMean",
    "CircR",
    "CircVar",
    # "ADJACENCY_ONLY",
    "AGGREGATE",
    "WEIGHTED_AGGREGATE",
]

NEIGHBORS = ("Count", "Neighbors", "NeighborIndices")

AGGREGATE = (
    "Min",
    "Median",
    "Max",
    "Quantile",
    "Mode",
    "Mean",
    "Std",
    "Var",
    "Sum",
    "CircMean",
    "CircR",
    "CircVar",
)

WEIGHTED_AGGREGATE = (
    "Mean",
    "Sum",
    "CircMean",
    "CircR",
)


# Dont mess with the return types as that confuses the numba jit compiler!
@nb.njit
def Count(arr: NDArray):
    return arr.size


@nb.njit
def Median(arr: NDArray) -> NDArray:
    if arr.size == 0:
        return np.nan
    return np.median(arr)


@nb.njit
def Max(arr: NDArray) -> NDArray:
    if arr.size == 0:
        return np.nan
    return np.max(arr)


@nb.njit
def Min(arr: NDArray) -> NDArray:
    if arr.size == 0:
        return np.nan
    return np.min(arr)


def _quantile_factory(*qs: float):
    for q in qs:

        def make_quantile(q_value=q):
            @nb.njit
            def quantile(arr):
                if arr.size == 0:
                    return np.nan
                return np.quantile(arr, q_value)

            quantile.__name__ = f"Q{q_value:.2f}"
            return quantile

        yield make_quantile()


def Quantile(q: float):
    return next(_quantile_factory(q))


@nb.njit
def _ValueCounts(arr: NDArray[np.integer]) -> list[tuple[int, int]]:
    counter = {}
    for e in arr.flat:
        if e == np.nan:
            continue
        elif e not in counter:
            counter[e] = 1
        else:
            counter[e] += 1
    return sorted(counter.items(), key=lambda x: (x[1], x[0]), reverse=True)


def Neighbors():
    return


def NeighborIndices():
    return


@nb.njit
def Mode(arr):
    if arr.size == 0:
        return np.nan
    return _ValueCounts(arr)[0][0]


@nb.njit
def Std(arr: NDArray) -> NDArray:
    if arr.size == 0:
        return np.nan
    return np.std(arr)


@nb.njit
def Var(arr: NDArray) -> NDArray:
    if arr.size == 0:
        return np.nan
    return np.var(arr)


@nb.njit
def Mean(arr: NDArray):
    if arr.size == 0:
        return np.nan
    return np.mean(arr)


@nb.njit
def Mean_(arr: NDArray, weights=None):
    if weights is None:
        return np.mean(arr)
    else:
        return np.mean(arr * weights) / np.sum(weights)


@nb.njit
def Sum(arr: NDArray) -> NDArray:
    return np.sum(arr)


@nb.njit
def Sum_(arr: NDArray, weights=None) -> NDArray:
    return np.sum(arr * weights)


# Circular aggregations based on pingouin (https://pingouin-stats.org/build/html/api.html#circular)
@nb.njit
def CircMean(angles, weights=None) -> float:
    if angles.size == 0:
        return np.nan
    if weights is None:
        return np.angle(np.sum(np.exp(angles * 1j)))
    else:
        return np.angle(np.sum(weights * np.exp(angles * 1j)))


@nb.njit
def CircR(angles, weights=None) -> float:
    if angles.size == 0:
        return np.nan
    if weights is None:
        return np.abs(np.sum(np.exp(angles * 1j))) / angles.size
    else:
        return np.abs(np.sum(weights * np.exp(angles * 1j))) / np.sum(weights)


@nb.njit
def CircVar(angles, weights=None) -> float:
    if angles.size == 0:
        return np.nan
    return 1 - CircR(angles=angles, weights=weights)
