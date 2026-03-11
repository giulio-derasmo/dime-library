# src/dime/filters/prf.py
#
# PRFFilter: importance = element-wise product of query embedding
# and the mean embedding of the top-k retrieved documents.
#
# Fully vectorized: all queries processed in one matrix operation.

from __future__ import annotations

import numpy as np
import pandas as pd

from filters.base import DimeFilter
from src.dime.importance import ImportanceMatrix
from src.memmap_interface import CorpusEncoding, QueriesEncoding


class PRFFilter(DimeFilter):
    """
    Computes dimension importance as:
        importance[q, d] = qemb[q, d] * mean(dembs[top_k_docs, d])

    The intuition: dimensions where the query and its top retrieved documents
    agree (both have large values) are likely the most discriminative.

    Args:
        qrys_encoder:  QueriesEncoding — encodes queries
        docs_encoder:  CorpusEncoding  — encodes documents
        run:           TREC-format run DataFrame (query_id, doc_id, rank, ...)
        k:             number of top documents to average
    """

    def __init__(
        self,
        qrys_encoder: QueriesEncoding,
        docs_encoder: CorpusEncoding,
        run: pd.DataFrame,
        k: int,
    ):
        self.qrys_encoder = qrys_encoder
        self.docs_encoder = docs_encoder
        self.run          = run
        self.k            = k

    def compute(self, queries: pd.DataFrame) -> ImportanceMatrix:
        query_ids = queries["query_id"].tolist()

        # ── query embeddings: [N, D] ───────────────────────────────────────────
        qembs = self.qrys_encoder.get_encoding(query_ids)           # [N, D]

        # ── mean top-k doc embeddings: [N, D] ─────────────────────────────────
        # Filter run to top-k docs for each query, then batch-retrieve embeddings.
        topk_run = (
            self.run[
                self.run["query_id"].isin(query_ids) &
                (self.run["rank"] < self.k)
            ]
            .copy()
        )

        # Preserve query order: for each query, fetch its top-k doc embeddings
        # and average them. Vectorized via groupby + numpy indexing.
        mean_dembs = np.zeros_like(qembs)                           # [N, D]

        # batch-fetch all needed doc embeddings in one call
        all_dids   = topk_run["doc_id"].tolist()
        all_dembs  = self.docs_encoder.get_encoding(all_dids)       # [M, D]

        # assign back per query
        topk_run = topk_run.reset_index(drop=True)
        topk_run["_emb_row"] = np.arange(len(topk_run))

        qid_to_row = {qid: i for i, qid in enumerate(query_ids)}
        for qid, group in topk_run.groupby("query_id"):
            if qid not in qid_to_row:
                continue
            emb_rows = group["_emb_row"].to_numpy()
            mean_dembs[qid_to_row[qid]] = all_dembs[emb_rows].mean(axis=0)

        # ── interaction: element-wise product ─────────────────────────────────
        scores = np.multiply(qembs, mean_dembs).astype(np.float32)  # [N, D]

        return ImportanceMatrix(scores=scores, query_ids=query_ids)

    @property
    def tag(self) -> str:
        return f"prf-k{self.k}"

    def __repr__(self) -> str:
        return f"PRFFilter(k={self.k})"