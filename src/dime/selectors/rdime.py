# src/dime/selectors/rdime.py
#
# RDIMESelector — Risk Dimension Importance Estimation (RDIME).
#
# Implements the hard-thresholding criterion from:
#   "Statistical Foundations of DIME: Risk Estimation for Practical
#    Index Selection" (D'Erasmo et al., EACL 2026).
#
# Core idea (Corollary 1 of the paper):
# ──────────────────────────────────────
# A query embedding q can be modelled as a noisy observation of a
# latent information need θ:
#
#     q = θ + ε·z,   z ~ N(0, I)
#
# The DIME importance vector u_q estimates θ² component-wise.
# The optimal hard-threshold estimator (Theorem 1) retains dimension i
# if and only if its estimated signal exceeds the estimated noise:
#
#     Ŝ = { i | (u_q)_i > ε̂² }
#
# where the noise level is estimated from the data as:
#
#     ε̂² = (1/D) · Σ_i ( q_i² − (u_q)_i )
#          = mean( q² − u_q )   per query
#
# This threshold is computed independently per query, so different
# queries retain different numbers of dimensions — no alpha needed.
#
# Contrast with TopAlphaSelector:
# ─────────────────────────────────────────────────────────────────────
#   TopAlphaSelector: keeps a fixed fraction α·D of dimensions
#                     (same budget for every query, requires grid search)
#   RDIMESelector:    keeps however many dimensions clear the noise bar
#                     (query-specific budget, zero hyperparameters)
#
# Interface note:
# ─────────────────────────────────────────────────────────────────────
# select() accepts alpha for API compatibility with DimSelector, but
# ignores it. Call search() or run_once() on MaskedSearcher instead
# of sweep() when using this selector — sweeping alpha is meaningless.

import numpy as np

from src.dime.selectors.base import DimSelector
from src.dime.importance import ImportanceMatrix


class RDIMESelector(DimSelector):
    """
    Hyperparameter-free hard-threshold selector based on risk estimation.

    For each query q with importance vector u_q, the per-query noise
    threshold is estimated as:

        ε̂²_q = mean_d( q_d² − (u_q)_d )

    A dimension d is retained if (u_q)_d > ε̂²_q, zeroed otherwise.

    Args:
        qembs: float32 array of shape [N, D] — raw query embeddings,
               aligned row-by-row with the ImportanceMatrix passed to
               select(). Needed because ImportanceMatrix stores u_q
               (the filter output) but not q itself.

    Usage:
        qembs    = qrys_encoder.get_encoding(query_ids)   # [N, D]
        selector = RDIMESelector(qembs)
        weights  = selector.select(importance, alpha=None) # alpha ignored
    """

    def __init__(self, qembs: np.ndarray):
        if qembs.ndim != 2:
            raise ValueError(f"qembs must be 2-D, got shape {qembs.shape}")
        self._qembs = qembs.astype(np.float32)   # [N, D]

    def select(self, importance: ImportanceMatrix, alpha: float = None) -> np.ndarray:
        """
        Compute a binary mask [N, D] using the RDIME threshold.

        alpha is accepted for interface compatibility but is ignored.
        The threshold is derived entirely from the importance scores
        and the query embeddings provided at construction time.

        Returns:
            float32 mask [N, D]: 1.0 for retained dimensions, 0.0 otherwise.
        """
        u_q = importance.scores                          # [N, D] float32 — DIME estimates of θ²

        # ── per-query noise estimate: ε̂²_q = mean_d(q_d² − (u_q)_d) ──────────
        eps2 = np.mean(self._qembs ** 2 - u_q, axis=1)  # [N] — one scalar per query

        # ── threshold: keep dimension d iff (u_q)_d > ε̂²_q ───────────────────
        # Broadcast eps2 [N] → [N, 1] for comparison against u_q [N, D]
        mask = (u_q > eps2[:, None])                     # [N, D] bool

        return mask.astype(np.float32)

    def mean_retained_frac(self, importance: ImportanceMatrix) -> float:
        """
        Compute the average fraction of dimensions retained across queries.
        Useful for logging and sanity-checking after selection.

        Returns a float in (0, 1].
        """
        weights = self.select(importance)
        return float(weights.mean())

    def retained_fracs(self, importance: ImportanceMatrix) -> np.ndarray:
        """
        Per-query fraction of retained dimensions. Shape [N].
        Useful for analysing the distribution of query-specific budgets.
        """
        weights = self.select(importance)
        return weights.mean(axis=1)                       # [N]

    @property
    def tag(self) -> str:
        return "rdime"

    def __repr__(self) -> str:
        return f"RDIMESelector(n_queries={self._qembs.shape[0]}, n_dims={self._qembs.shape[1]})"