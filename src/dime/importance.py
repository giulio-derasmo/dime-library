# src/dime/importance.py
#
# ImportanceMatrix — central data structure for DIME dimension filtering.
# Holds raw per-query importance scores over embedding dimensions and
# exposes fast mask generation for alpha-sweep experiments.

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass
class ImportanceMatrix:
    """
    Stores per-query importance scores over embedding dimensions.

    Attributes:
        scores:    float32 array of shape [N, D] — higher = more important
        query_ids: list of N query id strings, aligned with scores rows
    """

    scores:    np.ndarray        # [N, D] float32
    query_ids: list[str]

    # ── derived / cached ───────────────────────────────────────────────────────

    _query_to_row: dict[str, int] = field(init=False, repr=False)

    def __post_init__(self):
        if self.scores.ndim != 2:
            raise ValueError(f"scores must be 2-D, got shape {self.scores.shape}")
        if len(self.query_ids) != self.scores.shape[0]:
            raise ValueError(
                f"query_ids length ({len(self.query_ids)}) != scores rows ({self.scores.shape[0]})"
            )
        self._query_to_row = {qid: i for i, qid in enumerate(self.query_ids)}

    # ── shape helpers ──────────────────────────────────────────────────────────

    @property
    def n_queries(self) -> int:
        return self.scores.shape[0]

    @property
    def n_dims(self) -> int:
        return self.scores.shape[1]

    # ── mask generation ────────────────────────────────────────────────────────

    def top_k_mask(self, k: int) -> np.ndarray:
        """
        Return a boolean mask [N, D] that is True for the top-k dimensions
        per query (by importance score).

        Uses np.argpartition — O(N·D), avoids full sort.
        """
        if k >= self.n_dims:
            return np.ones_like(self.scores, dtype=bool)
        if k <= 0:
            return np.zeros_like(self.scores, dtype=bool)

        # argpartition gives us the k largest indices (unordered) per row
        top_indices = np.argpartition(self.scores, -k, axis=1)[:, -k:]   # [N, k]

        mask = np.zeros(self.scores.shape, dtype=bool)
        rows = np.arange(self.n_queries)[:, None]                         # [N, 1]
        mask[rows, top_indices] = True
        return mask

    def top_alpha_mask(self, alpha: float) -> np.ndarray:
        """
        Return a boolean mask [N, D] that is True for the top-(alpha * D)
        dimensions per query.

        alpha in (0, 1].  alpha=1.0 keeps all dimensions.
        """
        k = max(1, int(round(alpha * self.n_dims)))
        return self.top_k_mask(k)

    # ── subset helpers ─────────────────────────────────────────────────────────

    def subset(self, query_ids: Sequence[str]) -> "ImportanceMatrix":
        """Return a new ImportanceMatrix restricted to the given query_ids."""
        rows = [self._query_to_row[qid] for qid in query_ids]
        return ImportanceMatrix(
            scores=self.scores[rows],
            query_ids=list(query_ids),
        )

    def row(self, query_id: str) -> np.ndarray:
        """Return the importance vector [D] for a single query."""
        return self.scores[self._query_to_row[query_id]]

    # ── inspection / export ────────────────────────────────────────────────────

    def to_dataframe(self) -> pd.DataFrame:
        """
        Expand to a long DataFrame with columns: query_id, dim, importance, drank.
        Useful for debugging or saving; avoid in hot loops.
        """
        n, d  = self.scores.shape
        ranks = np.argsort(np.argsort(-self.scores, axis=1), axis=1) + 1  # [N, D]
        return pd.DataFrame({
            "query_id":   np.repeat(self.query_ids, d),
            "dim":        np.tile(np.arange(d), n),
            "importance": self.scores.ravel(),
            "drank":      ranks.ravel(),
        })

    def __repr__(self) -> str:
        return (
            f"ImportanceMatrix("
            f"n_queries={self.n_queries}, "
            f"n_dims={self.n_dims}, "
            f"dtype={self.scores.dtype})"
        )