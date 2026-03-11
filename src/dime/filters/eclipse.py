# src/dime/filters/eclipse.py
#
# EclipseWrapper — adds negative feedback subtraction on top of any base filter.
#
# The Eclipse formula is:
#
#   importance(Q) = lambda_pos * base_importance(Q) - lambda_neg * (Q ⊙ neg_mean(Q))
#
# where:
#   base_importance(Q)  [N, D]  raw importance matrix from the wrapped filter
#   neg_mean(Q)         [N, D]  mean embedding of the bottom-kneg docs per query
#   Q                   [N, D]  query embeddings
#
# The base filter must return its importance matrix WITHOUT any internal scalar
# weighting — all scaling is owned here via lambda_pos / lambda_neg.
#
# The negative feedback term is identical across all Eclipse variants:
#   neg_feedback[q] = q ⊙ mean(dembs[bottom_kneg_docs_for_q])
#
# Usage (in build_filter):
#   base    = PRFFilter(qrys_encoder=..., docs_encoder=..., run=..., k=2)
#   eclipse = EclipseWrapper(base, docs_encoder=..., run=..., kneg=5,
#                            lambda_pos=1.0, lambda_neg=0.5)
#
# Tag is derived automatically: "prf-k2" -> "prf-k2-eclipse"
import logging

import numpy as np
import pandas as pd

from src.dime.filters.base import DimeFilter
from src.dime.importance import ImportanceMatrix
from src.memmap_interface import CorpusEncoding

logger = logging.getLogger(__name__)


class EclipseWrapper(DimeFilter):
    """
    Wraps any DimeFilter and subtracts a negative feedback term.

    Args:
        base_filter:  any instantiated DimeFilter (PRFFilter, GPTFilter, ...)
        docs_encoder: CorpusEncoding — used to fetch bottom-kneg doc embeddings
        run:          TREC-format run DataFrame (query_id, doc_id, rank, ...)
        kneg:         number of bottom-ranked docs to use as negative feedback
        lambda_pos:   weight for the base (positive) importance term
        lambda_neg:   weight for the negative feedback term
    """

    def __init__(
        self,
        base_filter: DimeFilter,
        docs_encoder: CorpusEncoding,
        run: pd.DataFrame,
        kneg: int,
        lambda_pos: float = 1.0,
        lambda_neg: float = 0.5,
    ):
        self.base_filter  = base_filter
        self.docs_encoder = docs_encoder
        self.run          = run
        self.kneg         = kneg
        self.lambda_pos   = lambda_pos
        self.lambda_neg   = lambda_neg

    def compute(self, queries: pd.DataFrame) -> ImportanceMatrix:
        query_ids = queries["query_id"].tolist()

        # ── positive term: delegate entirely to the wrapped filter ─────────────
        base_result     = self.base_filter.compute(queries)
        base_importance = base_result.scores                        # [N, D]

        # ── query embeddings [N, D] — needed for neg interaction ──────────────
        qembs = self.base_filter.qrys_encoder.get_encoding(query_ids)  # [N, D]

        # ── negative term: bottom-kneg docs per query ──────────────────────────
        # Sort descending by rank (i.e. worst-ranked = highest rank value)
        # and take the last kneg docs for each query.
        botn_run = (
            self.run[self.run["query_id"].isin(query_ids)]
            .sort_values("rank", ascending=False)
            .groupby("query_id")
            .head(self.kneg)
        )

        # batch-fetch all needed doc embeddings in one call
        all_dids  = botn_run["doc_id"].tolist()
        all_dembs = self.docs_encoder.get_encoding(all_dids)        # [M, D]

        # compute per-query mean of bottom-kneg doc embeddings
        botn_run  = botn_run.reset_index(drop=True)
        botn_run["_emb_row"] = np.arange(len(botn_run))

        qid_to_row  = {qid: i for i, qid in enumerate(query_ids)}
        mean_negdoc = np.zeros_like(qembs)                          # [N, D]

        for qid, group in botn_run.groupby("query_id"):
            if qid not in qid_to_row:
                continue
            emb_rows = group["_emb_row"].to_numpy()
            mean_negdoc[qid_to_row[qid]] = all_dembs[emb_rows].mean(axis=0)

        # ── negative feedback interaction ──────────────────────────────────────
        neg_feedback = np.multiply(qembs, mean_negdoc)              # [N, D]

        # ── Eclipse combination ────────────────────────────────────────────────
        scores = (
            self.lambda_pos * base_importance
            - self.lambda_neg * neg_feedback
        ).astype(np.float32)                                        # [N, D]

        return ImportanceMatrix(scores=scores, query_ids=query_ids)

    # ── identity ───────────────────────────────────────────────────────────────

    @property
    def tag(self) -> str:
        return f"{self.base_filter.tag}-eclipse"

    def __repr__(self) -> str:
        return (
            f"EclipseWrapper("
            f"base={self.base_filter.tag}, "
            f"kneg={self.kneg}, "
            f"lambda_pos={self.lambda_pos}, "
            f"lambda_neg={self.lambda_neg})"
        )