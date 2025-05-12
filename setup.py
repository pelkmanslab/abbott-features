"""Setup script to cythonize .pyx file."""

import numpy
from Cython.Build import cythonize
from setuptools import Extension, find_packages, setup

# Use the correct relative path to the Cython file
cython_file = (
    "src/abbott_features/features/neighborhood/neighborhood_matrix_parallel.pyx"
)

# Define extension
extensions = [
    Extension(
        "abbott_features.features.neighborhood.neighborhood_matrix_parallel",
        [cython_file],
        include_dirs=[numpy.get_include()],
    )
]

# Setup configuration that works with build_meta backend
setup(
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    ext_modules=cythonize(extensions),
)
