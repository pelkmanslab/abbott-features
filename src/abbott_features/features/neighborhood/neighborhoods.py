"""Functions to compute radius, knn or touch adjacejcy matrices based

on KDTrees or label images in case of touch neighborhood.
"""

import functools
import inspect
import logging
import warnings
from collections.abc import Callable, Iterable, Sequence
from itertools import accumulate, product
from typing import Literal, ParamSpec, TypeAlias, TypeVar, cast

import networkx as nx
import ngio
import numpy as np
import polars as pl
import spatial_image as si
from attrs import asdict, frozen
from numpy.typing import NDArray
from scipy import spatial
from scipy.ndimage import map_coordinates
from scipy.sparse import csr_array
from sklearn.neighbors import NearestNeighbors

from abbott_features.features.label import get_position_and_orientation_features
from abbott_features.features.neighborhood import aggregation_functions
from abbott_features.features.neighborhood.neighborhood_aggregation import (
    aggregate_rows_csr,
    aggregate_table_csr,
    aggregate_weighted_table_csr,
)
from abbott_features.features.neighborhood.neighborhood_matrix_parallel import (
    weighted_anisotropic_touch_matrix,
)
from abbott_features.features.types import SpatialImage
from abbott_features.fractal_tasks.polars_utils import unnest_all_structs

logger = logging.getLogger(__name__)

CSRArray: TypeAlias = csr_array
AggFn: TypeAlias = Callable[[NDArray], NDArray]


@frozen
class TouchAdjacency:
    csr: CSRArray
    is_weighted: bool = True


@frozen
class DelaunayAdjacency:
    csr: CSRArray
    is_weighted: bool = False


@frozen
class NeighborhoodQuery:
    pass


@frozen
class AdjacencyQuery(NeighborhoodQuery):
    pass


@frozen
class DistanceQuery(NeighborhoodQuery):
    pass


@frozen
class KernelQuery(NeighborhoodQuery):
    pass


@frozen
class RadiusAdjacencyQuery(AdjacencyQuery):
    r: int
    self_loops: bool = False

    def __str__(self):
        return f"RAD:{self.r}{'s' if self.self_loops else ''}"


@frozen
class KnnAdjacencyQuery(AdjacencyQuery):
    k: int
    self_loops: bool = False

    def __str__(self):
        return f"KNN:{self.k}{'s' if self.self_loops else ''}"


@frozen
class RadiusDistanceQuery(DistanceQuery):
    r: float | int
    self_loops: bool = False

    def __str__(self):
        return f"RADd:{self.r}{'s' if self.self_loops else ''}"


@frozen
class KnnDistanceQuery(DistanceQuery):
    k: int
    self_loops: bool = False

    def __str__(self):
        return f"KNNd:{self.k}{'s' if self.self_loops else ''}"


@frozen
class DelaunayQuery(AdjacencyQuery):
    n_steps: int = 1
    self_loops: bool = False
    threshold: float | None = None

    def __str__(self):
        return (
            f"DELAUNAY{f':th={self.threshold}' if self.threshold is not None else ''}:"
            f"{self.n_steps}{'s' if self.self_loops else ''}"
        )


@frozen
class TouchQuery(AdjacencyQuery):
    n_steps: int = 1
    self_loops: bool = False
    threshold: float | None = None

    def __str__(self):
        return (
            f"TOUCH{f':th={self.threshold}' if self.threshold is not None else ''}:"
            f"{self.n_steps}{'s' if self.self_loops else ''}"
        )


@frozen
class RadiusKernelQuery(KernelQuery):
    r: float | int
    self_loops: bool = True
    function: str = "triangular"
    row_normalized: bool = True


@frozen
class KnnKernelQuery(KernelQuery):
    k: int
    self_loops: bool = True
    function: str = "triangular"
    row_normalized: bool = True


P = ParamSpec("P")
T = TypeVar("T")


class tuple_product:  # type: ignore
    def __init__(self, *decorator_args):
        self.decorator_args = decorator_args

    def __call__(self, func: Callable[P, T]) -> Callable[P, tuple[T, ...]]:
        @functools.wraps(func)
        def wrapped_func(*args, **kwargs):
            function_signature = inspect.signature(func).parameters
            adjusted_kwargs = {
                **{
                    k: v.default
                    for k, v in function_signature.items()
                    if v.default is not inspect._empty
                },  # default arguments from signature
                **dict(
                    zip(function_signature, args)
                ),  # warp positional arguments as kwargs
                **kwargs,
            }

            # package arguments in iterable if they're not already.
            for decorator_arg in self.decorator_args:
                if isinstance(adjusted_kwargs[decorator_arg], str) or not isinstance(
                    adjusted_kwargs[decorator_arg], Iterable
                ):
                    adjusted_kwargs[decorator_arg] = [adjusted_kwargs[decorator_arg]]

            # separate out arguments to iterate over, iterate, return tuple of results.
            out = []
            iter_args = {e: adjusted_kwargs.pop(e) for e in self.decorator_args}
            for e in product(*iter_args.values()):
                params = {**dict(zip(iter_args.keys(), e)), **adjusted_kwargs}
                out.append(func(**params))
            return tuple(out)

        return wrapped_func


@tuple_product("r", "self_loops")
def radius(
    r, self_loops=False, distance=False
) -> RadiusAdjacencyQuery | RadiusDistanceQuery:
    if distance:
        return RadiusDistanceQuery(r=r, self_loops=self_loops)
    return RadiusAdjacencyQuery(r=r, self_loops=self_loops)


@tuple_product("k", "self_loops")
def knn(k, self_loops=False, distance=False) -> KnnAdjacencyQuery | KnnDistanceQuery:
    if distance:
        return KnnDistanceQuery(k=k, self_loops=self_loops)
    return KnnAdjacencyQuery(k=k, self_loops=self_loops)


@tuple_product("r", "self_loops", "function")
def radius_kernel(
    r, self_loops=True, function="triangular", row_normalized=True
) -> RadiusKernelQuery:
    return RadiusKernelQuery(
        r=r, self_loops=self_loops, function=function, row_normalized=row_normalized
    )


@tuple_product("k", "self_loops", "function")
def knn_kernel(
    k, self_loops=True, function="triangular", row_normalized=True
) -> KnnKernelQuery:
    return KnnKernelQuery(
        k=k, self_loops=self_loops, function=function, row_normalized=row_normalized
    )


@tuple_product("n_steps", "self_loops", "threshold")
def delaunay(
    n_steps,
    self_loops=False,
    threshold=None,
) -> DelaunayQuery:
    return DelaunayQuery(n_steps=n_steps, self_loops=self_loops, threshold=threshold)


@tuple_product("n_steps", "self_loops", "threshold")
def touch(n_steps, self_loops=False, threshold=None) -> TouchQuery:
    return TouchQuery(n_steps=n_steps, self_loops=self_loops, threshold=threshold)


def _create_ranges(start, stop, N, endpoint=True):
    if endpoint == 1:
        divisor = N - 1
    else:
        divisor = N
    steps = (1.0 / divisor) * (stop - start)
    return steps[:, None] * np.arange(N) + start[:, None]


def multilineprofile(
    img: SpatialImage | NDArray,
    starts: NDArray,
    ends: NDArray,
    order: int = 3,
    n_samples: int = 30,
    scale: Sequence[int] | np.ndarray | None = None,
):
    """Interpolate an image between start- and end-points, i. e. compute lineprofiles.

    Points are assumed to be in physical coordinates! The image scale is taken from
    `SpatialImage.meta.scale` or can be provided manually using the scale argument.
    Be aware that samples on long lines are spaced further apart.


    Parameters
    ----------
    img : SpatialImage | NDArray
        Image from which to interpolate. If it's a `SpatialImage` the image scale is
        used to transform the points into pixel coordinates for interpolation.
    starts : NDArray (n_lines, n_dim)
        Start points of the lineprofiles, physical coordinates (SpatialImage),
        px coordinates (NDArray).
    ends : NDArray (n_lines, n_dim)
        End poitns of the lineprofile, physical coordinates (SpatialImage),
        px coordinates (NDArray).
    order : int, optional
        Order of interpolation (0-5, see `scipy.ndimage.map_coordinates`), by default 3
    n_samples : int, optional
        Number of samples to draw per lineprofile, by default 30
    scale: Sequence[int], np.ndarray, optional, (n_dim, )
        ([z], y, x) scale of the image. Passing scale overrides the scale read from a
        `SpatialImage`.

    Returns:
    -------
    NDArray (n_lines, n_samples)
        Each row corresponds to a lineprofile.
    """
    n_profiles = starts.shape[0]
    ndims = starts.shape[1]
    lines = np.array(
        [_create_ranges(starts[:, i], ends[:, i], n_samples) for i in range(ndims)]
    )
    lines = lines.reshape(ndims, -1)
    # TODO: Do this check with a validator (i. e. is_spatialimage)?
    if isinstance(img, SpatialImage) and scale is None:
        scale = np.asarray(img.meta.scale).reshape(1, -1).T
    elif scale is None:
        scale = np.array([1.0] * img.ndim).reshape(1, -1).T
    else:
        scale = np.array(scale).reshape(1, -1).T
    lines = lines / scale
    lineprofiles = map_coordinates(img, lines, order=order).reshape(
        n_profiles, n_samples
    )
    return lineprofiles


class make_iterable:  # type: ignore
    def __init__(self, *decorator_args):
        self.decorator_args = decorator_args

    def __call__(self, func):
        @functools.wraps(func)
        def wrapped_func(*args, **kwargs):
            adjusted_kwargs = {
                **dict(
                    zip(inspect.signature(func).parameters, args)
                ),  # warp positional arguments in a dict and pass as kwarg
                **kwargs,
            }
            for decorator_arg in self.decorator_args:
                if not isinstance(adjusted_kwargs[decorator_arg], Iterable):
                    adjusted_kwargs[decorator_arg] = [adjusted_kwargs[decorator_arg]]
            result = func(**adjusted_kwargs)
            return result

        return wrapped_func


class generator_over:  # type: ignore
    def __init__(self, *decorator_args):
        self.decorator_args = decorator_args

    def __call__(self, func):
        @functools.wraps(func)
        def wrapped_func(*args, **kwargs):
            adjusted_kwargs = {
                **dict(
                    zip(inspect.signature(func).parameters, args)
                ),  # warp positional arguments in a dict and pass as kwarg
                **kwargs,
            }

            # package arguments in iterable if they're not already.
            for decorator_arg in self.decorator_args:
                if not isinstance(adjusted_kwargs[decorator_arg], Iterable):
                    adjusted_kwargs[decorator_arg] = [adjusted_kwargs[decorator_arg]]

            # separate out arguments to iterate over, iterate, yield results.
            iter_args = {e: adjusted_kwargs.pop(e) for e in self.decorator_args}
            for e in product(*iter_args.values()):
                params = {**dict(zip(iter_args.keys(), e)), **adjusted_kwargs}
                yield func(**params)

        return wrapped_func


def _csr_matrix_power(csr: CSRArray, n: int) -> CSRArray:
    out = csr.copy()
    for _ in range(n - 1):
        out = out @ csr
    return out


def _csr_to_edge_indices(adj: CSRArray) -> NDArray:
    adj_coo = adj.tocoo()
    return np.vstack((adj_coo.row, adj_coo.col)).T


def _csr_concatenate_along_corner(arrs: CSRArray | Iterable[CSRArray]) -> CSRArray:
    if isinstance(arrs, CSRArray):
        arrs = [arrs]
    arrs = list(arrs)
    if len(arrs) == 1:
        return arrs[0]

    coo_arrays = [arr.tocoo() for arr in arrs]
    shapes = [arr.shape for arr in arrs]
    offsets = list(
        accumulate(
            shapes, lambda s0, s1: (s0[0] + s1[0], s0[1] + s1[1]), initial=(0, 0)
        )
    )
    out_shape = offsets[-1]

    row = np.concatenate(
        [arr.row + offset[0] for arr, offset in zip(coo_arrays, offsets)]
    )
    col = np.concatenate(
        [arr.col + offset[1] for arr, offset in zip(coo_arrays, offsets)]
    )
    data = np.concatenate([arr.data for arr in coo_arrays])

    # return data, indices, indptrs
    return csr_array((data, (row, col)), shape=out_shape)


def _delaunay_adjacency(
    points: NDArray,
) -> CSRArray:
    delaunay = spatial.Delaunay(points)
    G = nx.Graph()
    for path in delaunay.simplices:
        nx.add_path(G, path)
    return csr_array(nx.adjacency_matrix(G, nodelist=np.arange(points.shape[0])))


# TODO: Weighted by "shared surface"?
def get_delaunay_adjacency(
    points: NDArray,
    mask: SpatialImage | None = None,
    n_samples=50,
):
    adj = _delaunay_adjacency(points)
    if mask is None:
        return DelaunayAdjacency(csr=adj, is_weighted=False)

    adj_coo = adj.tocoo()
    start_idxs, end_idxs = adj_coo.row, adj_coo.col
    start, end = points[start_idxs], points[end_idxs]
    percentage_in_mask = multilineprofile(
        mask, start, end, order=0, n_samples=n_samples
    ).mean(axis=1)
    adj.data = percentage_in_mask
    return DelaunayAdjacency(csr=adj, is_weighted=True)


# TODO: implement
def get_touch_adjacency(label_image: SpatialImage, scale: tuple, absolute_surface=True):
    touch_matrix = weighted_anisotropic_touch_matrix(
        label_image.to_numpy().astype("int32"), *scale
    )
    np.fill_diagonal(touch_matrix, 0)
    labels = np.unique(label_image)[1:]
    if not absolute_surface:
        touch_matrix = touch_matrix / touch_matrix.sum(axis=1, keepdims=True)
    adj = csr_array(touch_matrix[labels][:, labels])
    return TouchAdjacency(csr=adj, is_weighted=True)


@generator_over("neighbors")
def query_radius_adjacency(
    neighbors: NearestNeighbors,
    r: float,
    self_loops: bool = False,
) -> CSRArray:
    X = neighbors._fit_X if self_loops else None

    query = neighbors.radius_neighbors_graph(
        X=X,
        radius=r,
        mode="connectivity",
    )
    return csr_array(query)


@generator_over("neighbors")
def query_radius_distance(
    neighbors: NearestNeighbors,
    r: float,
    self_loops: bool = False,
) -> CSRArray:
    X = neighbors._fit_X if self_loops else None

    query = neighbors.radius_neighbors_graph(
        X=X,
        radius=r,
        mode="distance",
    )
    return csr_array(query)


@generator_over("neighbors")
def query_knn_adjacency(
    neighbors: NearestNeighbors,
    k: int | Iterable[int],
    self_loops: bool = False,
) -> CSRArray:
    n_objects = neighbors._fit_X.shape[0]
    if k > (n_objects - 1):
        logger.warn(
            f"k={k} > (n_objects-1)={(n_objects-1)}; setting k to {n_objects-1}"
        )
        k = n_objects - 1
    X = neighbors._fit_X if self_loops else None
    query = neighbors.kneighbors_graph(
        X=X,
        n_neighbors=k + int(self_loops),  # Query k+1 neighbors in case of self-loops
        mode="connectivity",
    )
    return csr_array(query)


@generator_over("neighbors")
def query_knn_distance(
    neighbors: NearestNeighbors,
    k: int | Iterable[int],
    self_loops: bool = False,
) -> CSRArray:
    n_objects = neighbors._fit_X.shape[0]
    if k > (n_objects - 1):
        logger.warn(
            f"k={k} > (n_objects-1)={(n_objects-1)}; setting k to {n_objects-1}"
        )
        k = n_objects - 1
    X = neighbors._fit_X if self_loops else None
    query = neighbors.kneighbors_graph(
        X=X,
        n_neighbors=k + int(self_loops),
        mode="distance",
    )
    return csr_array(query)


def _kernel_triangular(z):
    return 1 - np.abs(z)


def _kernel_uniform(z):
    return np.full_like(z, 0.5)


def _kernel_gaussian(z):
    return (2 * np.pi) ** (-0.5) * np.exp(-(z**2) / 2)


KERNEL_FUNCTIONS = {
    "uniform": _kernel_uniform,
    "triangular": _kernel_triangular,
    "gaussian": _kernel_gaussian,
}


def _normalize_rows_csr(arr: CSRArray, inplace=False) -> CSRArray:
    row_weights = aggregate_rows_csr(arr, aggregation_functions.Sum, keepdims=False)
    distributed_row_weights = np.repeat(row_weights, np.diff(arr.indptr))
    if inplace:
        arr.data = arr.data / distributed_row_weights
        return arr
    else:
        return csr_array(
            (arr.data / distributed_row_weights, arr.indices, arr.indptr), arr.shape
        )


@generator_over("neighbors")
def query_radius_kernel(
    neighbors: NearestNeighbors,
    r: float,
    self_loops: bool = True,
    function: str = "triangular",
    row_normalized: bool = True,
) -> CSRArray:
    """i. e. fixed bandwidth"""
    distance_matrix = next(
        query_radius_distance(neighbors=neighbors, r=r, self_loops=self_loops)
    )
    z = distance_matrix.data / r
    # Adjust the weights according to the kernel function in-place
    distance_matrix.data = KERNEL_FUNCTIONS[function](z)
    # Apply row normalization
    if row_normalized:
        distance_matrix = _normalize_rows_csr(distance_matrix)
    return distance_matrix


@generator_over("neighbors")
def query_knn_kernel(
    neighbors: NearestNeighbors,
    k: int,
    self_loops: bool = True,
    function: str = "triangular",
    row_normalized: bool = True,
) -> CSRArray:
    """i. e. adaptive bandwidth"""
    n_dist, n_idx = neighbors.kneighbors(n_neighbors=k)
    bandwidth = n_dist[:, -1]
    distance_matrix = next(
        query_knn_distance(neighbors=neighbors, k=k, self_loops=self_loops)
    )
    z = distance_matrix.data / np.repeat(bandwidth, k + int(self_loops))
    distance_matrix.data = KERNEL_FUNCTIONS[function](z)
    if row_normalized:
        distance_matrix = _normalize_rows_csr(distance_matrix)
    return distance_matrix


# TODO: Delaunay distance queries?
def query_delaunay_adjacency(
    adj: CSRArray,
    n_steps: int,
    self_loops: bool = False,
    threshold: float | None = None,
) -> CSRArray:
    adj_out = adj.copy()

    if threshold is not None:
        adj_out.data = np.where(adj_out.data >= threshold, 1, 0)
        adj_out.eliminate_zeros()

    adj_out = _csr_matrix_power(adj_out, n_steps)
    adj_out.data = np.ones_like(adj_out.data)
    if self_loops:
        adj_out.setdiag(1)
    else:
        adj_out.setdiag(0)
        adj_out.eliminate_zeros()

    return adj_out


class NeighborhoodQueryObject:
    def __init__(
        self,
        neighbors: dict[str, NearestNeighbors],
        touch_adjacency: TouchAdjacency | None = None,
        delaunay_adjacency: DelaunayAdjacency | None = None,
        label: pl.DataFrame | None = None,
        queries: "tuple[NeighborhoodQuery, ...]" = (),
    ):
        self.neighbors = neighbors
        self.touch_adjacency = touch_adjacency
        self.delaunay_adjacency = delaunay_adjacency
        self.label = label
        self.queries = queries

    @classmethod
    def from_dataframe(
        cls,
        df: pl.DataFrame,
        label_columns: Sequence[str] | str | None = "label",
        centroid_column: Sequence[str] | str = "^Centroid(.[xyz])?$",
        region_id_column: str | None = None,
        touch_adjacency: TouchAdjacency | None = None,
        delaunay_adjacency: DelaunayAdjacency | None = None,
        neighbors_algorithm: Literal["auto", "ball_tree", "kd_tree", "brute"] = "auto",
    ) -> "NeighborhoodQueryObject":
        if label_columns is None:
            label = pl.Series("label", np.arange(len(df))).to_frame()
        else:
            if isinstance(label_columns, str):
                label_columns = [label_columns]
            for label_column in label_columns:
                assert label_column in df, f"'{label_column}' not found."
            label = df.select(label_columns)
            assert (
                label.unique().height == df.height
            ), f"None unique labels when using label_columns: `{tuple(label_columns)}`"

        if region_id_column is not None:
            assert (
                region_id_column in df
            ), f"`region_id_column` '{region_id_column}' not found."
            assert df[
                region_id_column
            ].is_sorted(), "`df` must be sorted by `region_id_column`."
            dfs = df.groupby(region_id_column, maintain_order=True)
        else:
            dfs = [("site_0", df)]

        neighbors = {}
        delaunays = []
        for site_name, df in dfs:
            if isinstance(centroid_column, str):
                df_centroid = df.select(centroid_column).pipe(unnest_all_structs)
            else:
                df_centroid = df.select(pl.col(e) for e in centroid_column)

            points = df_centroid.select(
                sorted(df_centroid.columns, reverse=True)
            ).to_numpy()

            neighbors[site_name] = NearestNeighbors(
                n_neighbors=5, radius=40, n_jobs=-1, algorithm=neighbors_algorithm
            ).fit(points)

            if delaunay_adjacency is None:
                delaunays.append(get_delaunay_adjacency(points))

        if delaunay_adjacency is None:
            delaunay_adjacency = DelaunayAdjacency(
                csr=_csr_concatenate_along_corner(e.csr for e in delaunays),
                is_weighted=False,
            )

        return NeighborhoodQueryObject(
            neighbors=neighbors,
            label=label,
            touch_adjacency=touch_adjacency,
            delaunay_adjacency=delaunay_adjacency,
        )

    @classmethod
    def from_labelimage(
        cls,
        label_image: ngio.images.label.Label | dict[str, ngio.images.label.Label],
        label_image_to: ngio.images.label.Label
        | dict[str, ngio.images.label.Label]
        | None = None,
        roi: ngio.common._roi.Roi | None = None,
        mask_n_samples: int = 100,
    ) -> "NeighborhoodQueryObject":
        axes_names = label_image.axes_mapper.on_disk_axes_names
        pixel_sizes = label_image.pixel_size.as_dict()
        scale = label_image.pixel_size.zyx

        label_numpy = label_image.get_roi(roi.name).astype("uint16")
        lbl = si.to_spatial_image(
            label_numpy,
            dims=axes_names,
            scale=pixel_sizes,
        )

        label_numpy_to = label_image_to.get_roi(roi).astype("uint16")
        mask = si.to_spatial_image(
            label_numpy_to,
            dims=axes_names,
            scale=pixel_sizes,
            name=label_image_to.meta.name,
        )

        if isinstance(lbl, dict):
            if mask is not None:
                assert isinstance(
                    mask, dict
                ), "You need to pass a dict of mask's if you pass a dict of lbl's"
                assert set(lbl.keys()) == set(
                    mask.keys()
                ), "Dictionaries have inconsistent keys."

            dfs = []
            delaunays = []
            for roi_id, lb in lbl.items():
                df = get_position_and_orientation_features(lb).with_columns(
                    roi=pl.lit(roi_id)
                )
                dfs.append(df)
                if mask is not None:
                    points = (
                        df.pipe(unnest_all_structs, sep=".")
                        .select(["Centroid.z", "Centroid.y", "Centroid.x"])
                        .to_numpy()
                    )
                    delaunays.append(
                        get_delaunay_adjacency(
                            points, mask[roi_id], n_samples=mask_n_samples
                        )
                    )
            df = pl.concat(dfs)
            delaunay_adjacency = DelaunayAdjacency(
                csr=_csr_concatenate_along_corner(e.csr for e in delaunays),
                is_weighted=True,
            )

        else:
            df = get_position_and_orientation_features(lbl)
            if mask is not None:
                points = (
                    df.pipe(unnest_all_structs, sep=".")
                    .select(["Centroid.z", "Centroid.y", "Centroid.x"])
                    .to_numpy()
                )
                delaunay_adjacency = get_delaunay_adjacency(
                    points, mask, n_samples=mask_n_samples
                )
            else:
                delaunay_adjacency = None
        touch_adjacency = get_touch_adjacency(lbl, scale, absolute_surface=False)

        return cls.from_dataframe(
            df, delaunay_adjacency=delaunay_adjacency, touch_adjacency=touch_adjacency
        )

    # TODO: functools.cached_property?
    @property
    def points_per_site(self):
        return {
            site_id: neighbors._fit_X for site_id, neighbors in self.neighbors.items()
        }

    @property
    def points(self):
        return np.concatenate([e._fit_X for e in self.neighbors.values()], axis=0)

    @property
    def index(self):
        return self.label.with_row_count(name="index").select(pl.col("index"))

    def add_queries(self, queries: "tuple[NeighborhoodQuery, ...]"):
        return NeighborhoodQueryObject(
            neighbors=self.neighbors,
            touch_adjacency=self.touch_adjacency,
            delaunay_adjacency=self.delaunay_adjacency,
            label=self.label,
            queries=self.queries + queries,
        )

    def radius(
        self,
        r: float | Iterable[float],
        self_loops: bool = False,
        distance: bool = False,
    ) -> "NeighborhoodQueryObject":
        return self.add_queries(radius(r=r, self_loops=self_loops, distance=distance))

    def knn(
        self, k: int | Iterable[int], self_loops: bool = False, distance: bool = False
    ) -> "NeighborhoodQueryObject":
        return self.add_queries(knn(k=k, self_loops=self_loops, distance=distance))

    def delaunay(
        self,
        n_steps: int | Iterable[int] = 1,
        self_loops: bool = False,
        threshold: float | None = None,
    ) -> "NeighborhoodQueryObject":
        return self.add_queries(
            delaunay(n_steps=n_steps, self_loops=self_loops, threshold=threshold)
        )

    def touch(
        self,
        n_steps: int | Iterable[int] = 1,
        self_loops: bool = False,
        threshold: float | None = None,
    ) -> "NeighborhoodQueryObject":
        return self.add_queries(
            touch(n_steps=n_steps, self_loops=self_loops, threshold=threshold)
        )

    # TODO: static method?
    def generate_neighborhood(self, query: NeighborhoodQuery) -> CSRArray:
        if isinstance(query, RadiusAdjacencyQuery):
            arrays = query_radius_adjacency(self.neighbors.values(), **asdict(query))
        elif isinstance(query, KnnAdjacencyQuery):
            arrays = query_knn_adjacency(self.neighbors.values(), **asdict(query))
        elif isinstance(query, RadiusDistanceQuery):
            arrays = query_radius_distance(self.neighbors.values(), **asdict(query))
        elif isinstance(query, KnnDistanceQuery):
            arrays = query_knn_distance(self.neighbors.values(), **asdict(query))
        elif isinstance(query, DelaunayQuery):
            if not self.delaunay_adjacency:
                raise RuntimeError(
                    "`NeighborhoodQueryObject.delauny_adjacency` not found!"
                )
            arrays = query_delaunay_adjacency(
                self.delaunay_adjacency.csr, **asdict(query)
            )
        elif isinstance(query, TouchQuery):
            if not self.touch_adjacency:
                raise RuntimeError(
                    "`NeighborhoodQueryObject.touch_adjacency` not found! Did you "
                    "instantiate NeighborhoodQueryObject.from_labelimage or provide "
                    "`touch_adjacency`?"
                )
            arrays = query_delaunay_adjacency(self.touch_adjacency.csr, **asdict(query))
        else:
            raise ValueError(f"Unknown `NeighborhoodQuery` {query}")
        return _csr_concatenate_along_corner(arrays)

    @property
    def neighborhoods(self) -> Iterable[CSRArray]:
        for query in self.queries:
            yield self.generate_neighborhood(query)

    @property
    def query_neighborhoods(
        self,
    ) -> Iterable[tuple[NeighborhoodQuery, CSRArray]]:
        for query in self.queries:
            yield query, self.generate_neighborhood(query)

    def aggregate(
        self,
        func: AggFn | Sequence[AggFn],
        df: pl.DataFrame | pl.Series | None = None,
        drop_label_columns_from_df: bool = True,
        return_label: bool = False,
        edge_weights: bool = False,
    ) -> pl.DataFrame:
        if not isinstance(func, Sequence):
            funcs = [cast(AggFn, func)]
        else:
            funcs = func

        if isinstance(df, pl.DataFrame):
            df = unnest_all_structs(df)
        elif isinstance(df, pl.Series):
            df = unnest_all_structs(df.to_frame())
        elif df is None:
            for func in funcs:
                assert (
                    func.__name__ in aggregation_functions.NEIGHBORS
                ), f"Cannot to '{func.__name__}' aggregation without `df`."
        else:
            raise ValueError(
                f"`df` must be polars.Series or polars.DataFrame, not `{type(df)}`"
            )

        if df is not None and all(c in df for c in self.label.columns):
            rows_before = df.height
            df = self.label.join(df, how="left", on=self.label.columns)
            if df.height != rows_before:
                warnings.warn(
                    "Joining `NeighborhoodQueryObject.label` with the provided `df` "
                    f"lead to dropping of {rows_before-df.height}/{rows_before} rows.",
                    stacklevel=2,
                )
            if drop_label_columns_from_df:
                for c in self.label.columns:
                    df.drop_in_place(c)

        results = []
        for query, adj in self.query_neighborhoods:
            for func in funcs:
                name = str(query)
                base_name = f"{name}_{func.__name__}"

                if func.__name__ == "Count":
                    aggregated_table = aggregate_table_csr(
                        adj, np.ones((adj.shape[0], 1)), func
                    )

                    results.append(
                        pl.DataFrame(
                            data=aggregated_table,
                            schema=[base_name],
                        ).select(pl.all().cast(pl.UInt32))
                    )

                elif func.__name__ == "Neighbors" or func.__name__ == "NeighborIndices":
                    if self.label is None:
                        # FIXME: is this necessary, or should label be enfoced
                        # upon creation?
                        raise ValueError(
                            "Provide a `meta` dataframe with `label` column."
                        )
                    results.append(
                        self.index.join(
                            pl.DataFrame(
                                _csr_to_edge_indices(adj),
                                schema={
                                    "index": pl.UInt32,
                                    "neighbor": pl.UInt32,
                                },
                            )
                            .groupby("index", maintain_order=True)
                            .agg(
                                pl.col("neighbor")
                                .map_dict(
                                    dict(
                                        zip(
                                            *self.label.select(
                                                pl.struct(pl.all())
                                            ).with_row_count()
                                        )
                                    )
                                )
                                .alias(base_name)
                                if func.__name__ == "Neighbors"
                                else pl.col("neighbor").alias(base_name)
                            ),
                            on="index",
                            how="left",
                        )
                        .drop("index")
                        .select(pl.all().fill_null([]))
                    )

                else:
                    prefix_str = f"{base_name}__"

                    if edge_weights:
                        aggregated_table = aggregate_weighted_table_csr(
                            adj, np.asarray(df), func
                        )
                    else:
                        aggregated_table = aggregate_table_csr(
                            adj, np.asarray(df), func
                        )

                    results.append(
                        pl.DataFrame(
                            data=aggregated_table,
                            schema=df.columns,
                        ).select(pl.all().prefix(prefix_str))
                    )

        if return_label:
            return self.label.with_columns(
                pl.concat(results, how="horizontal") if results else pl.DataFrame()
            )
        else:
            if results:
                return pl.concat(results, how="horizontal")
            else:
                return pl.DataFrame()

    def aggregate_weights(
        self,
        func: AggFn | Sequence[AggFn] = aggregation_functions.Sum,
        return_label: bool = False,
    ) -> pl.DataFrame:
        if not isinstance(func, Sequence):
            funcs = [cast(AggFn, func)]
        else:
            funcs = func

        results = []
        for query, adj in self.query_neighborhoods:
            for func in funcs:
                column_name = f"{query!s}_{func.__name__}"

                aggregated_weights = aggregate_rows_csr(adj, func)

                results.append(
                    pl.DataFrame(
                        data=aggregated_weights,
                        schema=[column_name],
                    )
                )

        if return_label:
            return self.label.with_columns(
                pl.concat(results, how="horizontal") if results else pl.DataFrame()
            )
        else:
            if results:
                return pl.concat(results, how="horizontal")
            else:
                return pl.DataFrame()
