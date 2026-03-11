# src/dime/filters/__init__.py

from src.dime.filters.base import DimeFilter
from src.dime.filters.prf import PRFFilter
from src.dime.filters.gpt import GPTFilter

# OracularFilter requires qrels so it is only imported when available
try:
    from src.dime.filters.oracular import OracularFilter
    __all__ = ["DimeFilter", "PRFFilter", "GPTFilter", "OracularFilter"]
except ImportError:
    __all__ = ["DimeFilter", "PRFFilter", "GPTFilter"]