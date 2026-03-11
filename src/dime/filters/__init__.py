# src/dime/filters/__init__.py

from src.dime.filters.base import DimeFilter
from src.dime.filters.prf import PRFFilter

# OracularFilter is imported lazily in the pipeline to avoid requiring
# qrels at import time, but it is exported here for direct library use.
try:
    from src.dime.filters.oracular import OracularFilter
    __all__ = ["DimeFilter", "PRFFilter", "OracularFilter"]
except ImportError:
    __all__ = ["DimeFilter", "PRFFilter"]