# src/dime/filters/gpt.py
#
# GPTFilter: importance = element-wise product of the query embedding
# and the embedding of a GPT-generated answer for that query.
#
# Intuition: dimensions where the query and a fluent, knowledge-rich
# answer agree are likely the most semantically discriminative.
# Identical interaction function to PRFFilter — only the source of the
# second vector differs (GPT text instead of mean top-k doc embeddings).
#
# Input CSV (one file per collection, wide format):
#   data/gpt/{collection}/{variant}.csv
#   columns: query_id (str), text (str)
#
# Queries missing from the CSV fall back to the query embedding itself,
# making the importance vector q ⊙ q = q² (all positive, safe fallback).

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.config import GPT_DIR, MODEL_TO_HF
from src.dime.filters.base import DimeFilter
from src.dime.importance import ImportanceMatrix
from src.memmap_interface import QueriesEncoding

logger = logging.getLogger(__name__)


class GPTFilter(DimeFilter):
    """
    Computes dimension importance as:
        importance[q, d] = qemb[q, d] * encode(gpt_answer[q])[d]

    The GPT answer is encoded with the same model used for queries and
    documents so that the embedding spaces are aligned.

    Args:
        qrys_encoder: QueriesEncoding — provides raw query embeddings
        model_name:   model key from config (e.g. "contriever") — used to
                      load the SentenceTransformer encoder for GPT answers
        collection:   collection key (e.g. "dl19") — selects the CSV file
        variant:      filename stem of the CSV (default: "gpt")
                      e.g. variant="gpt-k10" loads data/gpt/{collection}/gpt-k10.csv
    """

    def __init__(
        self,
        qrys_encoder: QueriesEncoding,
        model_name: str,
        collection: str,
        variant: str = "gpt",
    ):
        self.qrys_encoder = qrys_encoder
        self.collection   = collection
        self.variant      = variant

        # ── load GPT answers ───────────────────────────────────────────────────
        csv_path = GPT_DIR / collection / f"{variant}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"GPT answers not found at {csv_path}. "
                f"Expected columns: query_id, text."
            )
        answers = pd.read_csv(csv_path, dtype={"query_id": str})
        # keep first occurrence if duplicates exist, index by query_id for O(1) lookup
        self._answers: pd.Series = (
            answers
            .loc[~answers["query_id"].duplicated()]
            .set_index("query_id")["text"]
        )
        logger.info(f"Loaded {len(self._answers)} GPT answers from {csv_path}")

        # ── load encoder (same model as queries/docs for aligned embedding space) ──
        if model_name not in MODEL_TO_HF:
            raise ValueError(f"Unknown model '{model_name}'. Available: {list(MODEL_TO_HF)}")
        logger.info(f"Loading encoder for GPT answers: {model_name}")
        self._encoder = SentenceTransformer(
            MODEL_TO_HF[model_name],
            model_kwargs={"use_safetensors": True},
        )

    def compute(self, queries: pd.DataFrame) -> ImportanceMatrix:
        query_ids = queries["query_id"].tolist()

        # ── query embeddings [N, D] ────────────────────────────────────────────
        qembs = self.qrys_encoder.get_encoding(query_ids)   # [N, D]

        # ── encode GPT answers — batch encode all at once ──────────────────────
        # For queries missing from the CSV, substitute the query text itself
        # so the fallback importance is q ⊙ q = q² (all-positive, safe).
        missing = [qid for qid in query_ids if qid not in self._answers.index]
        if missing:
            logger.warning(
                f"{len(missing)} queries have no GPT answer and will use "
                f"the query embedding as fallback: {missing[:5]}"
                f"{'...' if len(missing) > 5 else ''}"
            )

        texts = [
            self._answers[qid] if qid in self._answers.index
            else queries.loc[queries["query_id"] == qid, "text"].iloc[0]
            for qid in query_ids
        ]

        aembs = self._encoder.encode(
            texts,
            batch_size=256,
            show_progress_bar=True,
            convert_to_numpy=True,
        )                                                    # [N, D]

        # ── interaction: element-wise product ─────────────────────────────────
        scores = np.multiply(qembs, aembs).astype(np.float32)   # [N, D]

        return ImportanceMatrix(scores=scores, query_ids=query_ids)

    @property
    def tag(self) -> str:
        # variant already encodes what this filter is, e.g. "gpt", "gpt-k10"
        return self.variant

    def __repr__(self) -> str:
        return f"GPTFilter(collection={self.collection}, variant={self.variant})"