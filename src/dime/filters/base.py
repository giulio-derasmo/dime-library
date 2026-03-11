# src/dime/filters/base.py
#
# Abstract base class for all dimension filters.
# Subclasses implement compute() to produce an ImportanceMatrix.
from abc import ABC, abstractmethod

import pandas as pd

from src.dime.importance import ImportanceMatrix


class DimeFilter(ABC):
    """
    Base class for dimension importance filters.

    A DimFilter takes a set of queries and produces an ImportanceMatrix —
    a [N_queries, D] float array where higher values indicate more important
    embedding dimensions for that query.

    All computation must be batched/vectorized over the full query set.
    No per-query Python loops in subclasses.

    Subclasses must implement:
        compute(queries) → ImportanceMatrix
        tag              → str used in saved filenames, e.g. "topk-k10"
    """

    @abstractmethod
    def compute(self, queries: pd.DataFrame) -> ImportanceMatrix:
        """
        Compute dimension importance for all queries.

        Args:
            queries: DataFrame with at minimum a 'query_id' column.

        Returns:
            ImportanceMatrix of shape [N_queries, D].
        """
        ...

    @property
    @abstractmethod
    def tag(self) -> str:
        """
        Short identifier used in saved filenames.
        Must be filesystem-safe (no spaces or slashes).
        e.g. "topk-k10", "oracular", "gpt", "identity"
        """
        ...

    def __call__(self, queries: pd.DataFrame) -> ImportanceMatrix:
        """Shorthand: filter(queries) == filter.compute(queries)."""
        return self.compute(queries)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(tag={self.tag})"