"""Functions to aggregate feature tables across neighborhoods represented by adjacency

matrices and unsing custom aggregation functions.
"""

from typing import Callable

import numba as nb
import numpy as np
import polars as pl
from scipy import sparse


def aggregate_table_nan(
    adjacency_matrix, feature_table, nan_skipping_aggregaction_function
):
    return np.apply_along_axis(
        nan_skipping_aggregaction_function,
        1,
        (
            np.expand_dims(adjacency_matrix, axis=-1)
            * np.expand_dims(feature_table, axis=0)
        ),
    )


@nb.jit(parallel=True, cache=False)
def aggregate_column_dense_parallel(
    adjacency_matrix: np.ndarray,
    feature_column: np.ndarray,
    aggregation_function: Callable,
) -> np.ndarray:
    neighborhood_features = np.zeros_like(feature_column)
    for observation_index in nb.prange(adjacency_matrix.shape[0]):
        adjacency_array = adjacency_matrix[observation_index, :]
        adjacency_mask = adjacency_array > 0

        adjacent_features = np.atleast_2d(feature_column[adjacency_mask])
        neighborhood_features[observation_index] = aggregation_function(
            adjacent_features
        )
    return neighborhood_features


@nb.njit(parallel=True)
def aggregate_table_dense_parallel(
    adjacency_matrix: np.ndarray,
    feature_array: np.ndarray,
    aggregation_function: Callable,
) -> np.ndarray:
    neighborhood_features = np.zeros_like(feature_array)

    for var_index in nb.prange(feature_array.shape[1]):
        feature_column = feature_array[:, var_index]
        neighborhood_features[:, var_index] = aggregate_column_dense_parallel(
            adjacency_matrix=adjacency_matrix,
            feature_column=feature_column,
            aggregation_function=aggregation_function,
        )
    return neighborhood_features


@nb.jit(parallel=True, cache=False)
def aggregate_weighted_column_dense_parallel(
    weight_matrix: np.ndarray,
    feature_column: np.ndarray,
    aggregation_function: Callable,
) -> np.ndarray:
    """Non-edges need to be np.nan!"""
    neighborhood_features = np.zeros_like(feature_column)
    for observation_index in nb.prange(weight_matrix.shape[0]):
        adjacency_array = weight_matrix[observation_index, :]
        adjacency_mask = ~np.isnan(adjacency_array)

        adjacent_features = np.atleast_2d(
            feature_column[adjacency_mask] * adjacency_array[adjacency_mask]
        )
        neighborhood_features[observation_index] = aggregation_function(
            adjacent_features
        )
    return neighborhood_features


@nb.njit(parallel=True)
def aggregate_weighted_table_dense_parallel(
    weight_matrix: np.ndarray,
    feature_array: np.ndarray,
    aggregation_function: Callable,
) -> np.ndarray:
    """Non-edges need to be np.nan!"""
    neighborhood_features = np.zeros_like(feature_array)
    for var_index in nb.prange(feature_array.shape[1]):
        feature_column = feature_array[:, var_index]
        neighborhood_features[:, var_index] = aggregate_weighted_column_dense_parallel(
            weight_matrix=weight_matrix,
            feature_column=feature_column,
            aggregation_function=aggregation_function,
        )
    return neighborhood_features


@nb.jit(parallel=True, cache=True, nopython=True)
def aggregate_rows(
    adjacency_matrix: np.ndarray,
    aggregation_function,
) -> np.ndarray:
    neighborhood_features = np.zeros((adjacency_matrix.shape[0], 1))

    for observation_index in nb.prange(adjacency_matrix.shape[0]):
        adjacency_array = adjacency_matrix[observation_index, :]
        adjacency_mask = ~np.isnan(adjacency_array)

        adjacent_features = adjacency_array[adjacency_mask]
        neighborhood_features[observation_index] = aggregation_function(
            adjacent_features
        )
    return neighborhood_features


@nb.jit(parallel=True, cache=False, nopython=True)
def aggregate_rows_csr_(
    indptr: np.ndarray,
    data: np.ndarray,
    aggregation_function,
) -> np.ndarray:
    neighborhood_features = np.zeros((indptr.shape[0] - 1, 1))

    for i in nb.prange(indptr.shape[0] - 1):
        adjacent_features = data[indptr[i] : indptr[i + 1]]
        neighborhood_features[i] = aggregation_function(adjacent_features)
    return neighborhood_features


@nb.jit(parallel=True, cache=False, nopython=True)
def distribute_feature_csr_(
    indices: np.ndarray,
    feature: np.ndarray,
) -> np.ndarray:
    data = np.zeros(indices.shape[0])

    for i in nb.prange(indices.shape[0]):
        data[i] = feature[indices[i], 0]
    return data


@nb.jit(parallel=True, cache=False, nopython=True)
def aggregate_frows_csr_(
    indptr: np.ndarray,
    data: np.ndarray,
    aggregation_function,
) -> np.ndarray:
    neighborhood_features = np.zeros(indptr.shape[0] - 1)

    for i in nb.prange(indptr.shape[0] - 1):
        adjacent_features = data[indptr[i] : indptr[i + 1]]
        neighborhood_features[i] = aggregation_function(adjacent_features)
    return neighborhood_features


@nb.jit(parallel=True, cache=False, nopython=True)
def distribute_ffeature_csr_(
    indices: np.ndarray,
    feature: np.ndarray,
) -> np.ndarray:
    data = np.zeros(indices.shape[0])

    for i in nb.prange(indices.shape[0]):
        data[i] = feature[indices[i]]
    return data


@nb.jit(parallel=True, cache=False, nopython=True)
def aggregate_table_csr_(
    indptr: np.ndarray,
    indices: np.ndarray,
    weights: np.ndarray,
    features: np.ndarray,
    aggregation_function,
) -> np.ndarray:
    neighborhood_features = np.zeros_like(features)
    for var_index in nb.prange(features.shape[1]):
        feature = features[:, var_index]
        data = distribute_ffeature_csr_(indices, feature)
        neighborhood_feature = aggregate_frows_csr_(indptr, data, aggregation_function)
        neighborhood_features[:, var_index] = neighborhood_feature
    return neighborhood_features


@nb.jit(parallel=True, cache=False, nopython=True)
def aggregate_weighted_table_csr_(
    indptr: np.ndarray,
    indices: np.ndarray,
    weights: np.ndarray,
    features: np.ndarray,
    aggregation_function,
) -> np.ndarray:
    neighborhood_features = np.zeros_like(features)
    for var_index in nb.prange(features.shape[1]):
        feature = features[:, var_index]
        data = distribute_ffeature_csr_(indices, feature) * weights
        neighborhood_feature = aggregate_frows_csr_(indptr, data, aggregation_function)
        neighborhood_features[:, var_index] = neighborhood_feature
    return neighborhood_features


@nb.jit(parallel=False, cache=False, nopython=True)
def aggregate_table_csr_nopar_(
    indptr: np.ndarray,
    indices: np.ndarray,
    weights: np.ndarray,
    features: np.ndarray,
    aggregation_function,
) -> np.ndarray:
    neighborhood_features = np.zeros_like(features)
    for var_index in nb.prange(features.shape[1]):
        feature = features[:, var_index]
        data = distribute_ffeature_csr_(indices, feature)
        neighborhood_feature = aggregate_frows_csr_(indptr, data, aggregation_function)
        neighborhood_features[:, var_index] = neighborhood_feature
    return neighborhood_features


def distribute_feature_csr(
    adj: sparse.csr_array,
    feature: np.ndarray,
) -> sparse.csr_array:
    return sparse.csr_array(
        (distribute_feature_csr_(adj.indices, feature), adj.indices, adj.indptr),
        shape=adj.shape,
    )


def aggregate_rows_csr(
    adj: sparse.csr_array,
    aggregation_function,
    keepdims=True,
) -> np.ndarray:
    if keepdims:
        return aggregate_rows_csr_(adj.indptr, adj.data, aggregation_function)
    else:
        return aggregate_frows_csr_(adj.indptr, adj.data, aggregation_function)


def aggregate_csr_slow(
    adj: sparse.csr_array,
    feature: np.ndarray,
    aggregation_function,
) -> np.ndarray:
    adj_w = adj * sparse.csr_array(feature.T)
    return aggregate_rows_csr(adj_w, aggregation_function)


def aggregate_csr_single(
    adj: sparse.csr_array,
    feature: np.ndarray,
    aggregation_function,
) -> np.ndarray:
    adj_w = distribute_feature_csr(adj, feature)
    return aggregate_rows_csr(adj_w, aggregation_function)


def aggregate_table_csr(
    adj: sparse.csr_array,
    features: np.ndarray,
    aggregation_function,
) -> np.ndarray:
    return aggregate_table_csr_(
        adj.indptr, adj.indices, adj.data, features, aggregation_function
    )


def aggregate_weighted_table_csr(
    adj: sparse.csr_array,
    features: np.ndarray,
    aggregation_function,
) -> np.ndarray:
    return aggregate_weighted_table_csr_(
        adj.indptr, adj.indices, adj.data, features, aggregation_function
    )


def aggregate_csr_nopar(
    adj: sparse.csr_array,
    features: np.ndarray,
    aggregation_function,
) -> np.ndarray:
    return aggregate_table_csr_nopar_(
        adj.indptr, adj.indices, adj.data, features, aggregation_function
    )


# %% Slower for my kind of data (<10_000 objects)
def aggregate_column_csr_polars(
    adjacency_csr: "sparse.csr_array",
    feature_column: np.ndarray | pl.Series,
    aggregation_function: pl.Expr = None,
    weighted: bool = False,
    maintain_order: bool = True,
    return_index: bool = False,
):
    if aggregation_function is None:
        aggregation_function = pl.col("feature").mean().name.prefix("Mean_")

    coo_array = adjacency_csr.tocoo(copy=False)
    if weighted:
        res = (
            pl.LazyFrame(
                [
                    pl.Series("focal", coo_array.row, dtype=pl.UInt32),
                    pl.Series("neighbor", coo_array.col, dtype=pl.UInt32),
                    pl.Series("weight", coo_array.data),
                ]
            )
            .join(
                pl.Series("feature", feature_column).to_frame().with_row_count().lazy(),
                left_on="neighbor",
                right_on="row_nr",
            )
            .with_columns(pl.col("feature") * pl.col("weight"))
            .groupby("focal", maintain_order=maintain_order)
            .agg(aggregation_function)
        )
    else:
        res = (
            pl.LazyFrame(
                [
                    pl.Series("focal", coo_array.row, dtype=pl.UInt32),
                    pl.Series("neighbor", coo_array.col, dtype=pl.UInt32),
                ]
            )
            .join(
                pl.Series("feature", feature_column).to_frame().with_row_count().lazy(),
                left_on="neighbor",
                right_on="row_nr",
            )
            .with_columns(pl.col("feature"))
            .groupby("focal", maintain_order=maintain_order)
            .agg(aggregation_function)
            # .sort(by='focal')
            # .select(pl.exclude('focal'))
        )
    if return_index:
        return res.collect()
    else:
        return res.select(pl.exclude("focal")).collect()


def aggregate_table_csr_polars(
    adjacency_csr: "sparse.csr_array",
    feature_array: np.ndarray | pl.DataFrame,
    aggregation_function: pl.Expr = None,
    weighted: bool = False,
    maintain_order: bool = True,
    return_index: bool = False,
):
    if aggregation_function is None:
        aggregation_function = pl.all().mean().name.prefix("Mean_")

    coo_array = adjacency_csr.tocoo(copy=False)
    if weighted:
        res = (
            pl.LazyFrame(
                [
                    pl.Series("focal", coo_array.row, dtype=pl.UInt32),
                    pl.Series("neighbor", coo_array.col, dtype=pl.UInt32),
                    pl.Series("weight", coo_array.data),
                ]
            )
            .join(
                pl.DataFrame(np.asarray(feature_array)).with_row_count().lazy(),
                left_on="neighbor",
                right_on="row_nr",
            )
            .select(
                pl.col("focal"),
                pl.exclude(["focal", "neighbor", "weight"]) * pl.col("weight"),
            )
            .groupby("focal", maintain_order=maintain_order)
            .agg(aggregation_function)
        )
    else:
        res = (
            pl.LazyFrame(
                [
                    pl.Series("focal", coo_array.row, dtype=pl.UInt32),
                    pl.Series("neighbor", coo_array.col, dtype=pl.UInt32),
                ]
            )
            .join(
                pl.DataFrame(np.asarray(feature_array)).with_row_count().lazy(),
                left_on="neighbor",
                right_on="row_nr",
            )
            .select(pl.col("focal"), pl.exclude(["focal", "neighbor", "weight"]))
            .groupby("focal", maintain_order=maintain_order)
            .agg(aggregation_function)
            # .sort(by='focal')
            # .select(pl.exclude('focal'))
        )
    if return_index:
        return res.collect()
    else:
        return res.select(pl.exclude("focal")).collect()
