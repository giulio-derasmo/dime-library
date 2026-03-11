# src/dime/selectors/base.py
#
# Abstract base class for dimension selection strategies.
#
# A DimSelector is responsible for one thing only:
# given an ImportanceMatrix and an alpha scalar, produce a weight
# matrix [N, D] that will be element-wise multiplied with the raw
# query embeddings before FAISS search.
#
# This decouples *how importance is computed* (the DimeFilter's job)
# from *how importance scores are turned into embedding weights*
# (the DimSelector's job).
#
# Hard masks (binary {0, 1}) and soft weights (continuous floats)
# are both valid — FAISS inner-product search handles both correctly.

from abc import ABC, abstractmethod

import numpy as np

from src.dime.importance import ImportanceMatrix


class DimSelector(ABC):
    """
    Base class for dimension selection strategies.

    A DimSelector maps (ImportanceMatrix, alpha) → weight matrix [N, D].

    The weight matrix is float32 and element-wise multiplied with the
    raw query embeddings before search:
        masked_q[n, d] = q[n, d] * weights[n, d]

    Subclasses must implement:
        select(importance, alpha) → np.ndarray  [N, D] float32
        tag                       → str          filesystem-safe identifier
    """

    @abstractmethod
    def select(self, importance: ImportanceMatrix, alpha: float) -> np.ndarray:
        """
        Compute a weight matrix from importance scores.

        Args:
            importance: ImportanceMatrix of shape [N, D]
            alpha:      float in (0, 1] controlling selection budget

        Returns:
            np.ndarray of shape [N, D], dtype float32
            Binary {0, 1} for hard masks; continuous values for soft weighting.
        """
        ...

    @property
    @abstractmethod
    def tag(self) -> str:
        """
        Short, filesystem-safe identifier used in saved filenames.
        e.g. "top-alpha", "soft-exp", "threshold-0.5"
        Must not contain spaces or slashes.
        """
        ...

    def __call__(self, importance: ImportanceMatrix, alpha: float) -> np.ndarray:
        """Shorthand: selector(importance, alpha) == selector.select(importance, alpha)."""
        return self.select(importance, alpha)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(tag={self.tag})"