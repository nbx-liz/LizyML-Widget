"""LizyML Widget - Jupyter/Colab/VS Code notebook widget for LizyML."""

from importlib.metadata import PackageNotFoundError, version

from .widget import LizyWidget

try:
    __version__ = version("lizyml-widget")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = ["LizyWidget", "__version__"]
