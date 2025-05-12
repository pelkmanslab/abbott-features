"""Build script for Cython files"""

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

extensions = [
    Extension(
        "abbott_features.features.neighborhood.neighborhood_matrix_parallel",
        ["src/abbott_features/features/neighborhood/neighborhood_matrix_parallel.pyx"],
        include_dirs=[np.get_include()],
    )
]

setup(
    ext_modules=cythonize(extensions),
)
