"""
Package description.
"""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("abbott-features")
except PackageNotFoundError:
    __version__ = "uninstalled"