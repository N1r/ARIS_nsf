"""ARIS: glass-box neural source-filter synthesis for phonetic stimulus manipulation."""

from aris.edit import edit, load, resynthesize
from aris.model import ARIS

__version__ = "1.0.0"
__all__ = ["ARIS", "edit", "load", "resynthesize"]
