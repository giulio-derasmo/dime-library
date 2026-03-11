# src/dime/selectors/top_alpha.py
#
# TopAlphaSelector: keeps the top (alpha * D) dimensions per query,
# zeros out the rest. This is the original DIME masking strategy.

import numpy as np

from src.dime.selectors.base import DimSelector
from src.dime.importance import ImportanceMatrix


class TopAlphaSelector(DimSelector):
    """
    Hard binary mask: keep the top-(alpha * D) dimensions per query,
    set the rest to zero.

    This is the original DIME strategy, extracted from ImportanceMatrix
    into a standalone selector so it can be swapped out cleanly.

    alpha=1.0 keeps all dimensions (equivalent to unmasked search).
    alpha=0.1 keeps only the top 10% of dimensions per query.
    """

    def select(self, importance: ImportanceMatrix, alpha: float) -> np.ndarray:
        """
        Returns a binary float32 mask [N, D]:
            1.0  for the top-(alpha * D) dimensions per query
            0.0  for all other dimensions

        Uses np.argpartition — O(N·D), avoids a full sort.
        """
        mask = importance.top_alpha_mask(alpha)   # [N, D] bool
        return mask.astype(np.float32)

    @property
    def tag(self) -> str:
        return "top-alpha"