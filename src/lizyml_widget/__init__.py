"""LizyML Widget - Jupyter/Colab/VS Code notebook widget for LizyML."""

from .widget import LizyWidget

try:
    from ._version import __version__
except ImportError:
    __version__ = "0.0.0"

__all__ = ["LizyWidget", "__version__"]
