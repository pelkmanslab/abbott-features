#cython: language_level=3

import numpy as np
cimport cython
from cython.parallel import prange


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.cdivision(True)
def weighted_anisotropic_touch_matrix(int[:, :, :] arr, float sz=1.0, float sy=1.0, float sx=1.0):
    """Compute an adjacency matrix on a (anisotropic) label image.

    Args:
        arr: label image.
        sz: scale along z. Defaults to 1.0.
        sy: scale along y. Defaults to 1.0.
        sx: sacle along x. Defaults to 1.0.
    Returns:
        adjacency matrix: np.array (max_label+1, max_label+1)
            with entries corresponding to the touching surface area.
    """    
    cdef int *index_shifts = [-1, 1]
    cdef Py_ssize_t z_max = arr.shape[0]
    cdef Py_ssize_t y_max = arr.shape[1]
    cdef Py_ssize_t x_max = arr.shape[2]
    cdef Py_ssize_t max_label = np.max(arr)
    cdef int v, vn
    cdef Py_ssize_t x, y, z, nx, ny, nz, n
    cdef float area_z = sy * sx
    cdef float area_y = sx * sz
    cdef float area_x = sy * sz

    neighborhood_matrix = np.zeros((max_label+1, max_label+1), dtype=np.float32)
    cdef float[:, :] result_view = neighborhood_matrix

    for z in prange(z_max, nogil=True):
        for y in range(y_max):
            for x in range(x_max):
                v = arr[z, y, x]
                for n in range(2):
                    nx = x + index_shifts[n]
                    if (nx>=0) and (nx<x_max):
                        vn = arr[z, y, nx]
                        result_view[v, vn] += area_x
                for n in range(2):
                    ny = y + index_shifts[n]
                    if (ny>=0) and (ny<y_max):
                        vn = arr[z, ny, x]
                        result_view[v, vn] += area_y
                for n in range(2):
                    nz = z + index_shifts[n]
                    if (nz>=0) and (nz<z_max):
                        vn = arr[nz, y, x]
                        result_view[v, vn] += area_z
    return neighborhood_matrix