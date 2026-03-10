# src/search.py

import logging

import numpy as np
import pandas as pd

from src.config import RUNS_DIR, get_corpus_name
from src.index import load_index
from src.memmap_interface import CorpusEncoding, QueriesEncoding

logger = logging.getLogger(__name__)


def search(
    model_name: str,
    collection: str,
    k: int = 1000,
) -> pd.DataFrame:
    """
    Search the FAISS index with all queries for a given collection.

    Requires:
      - encode_corpus()  to have been run
      - encode_queries() to have been run
      - build_index()    to have been run

    Returns a run dataframe in TREC format:
        query_id | Q0 | doc_id | rank | score | run
    """

    # ── load index and encodings ───────────────────────────────────────────────
    logger.info(f"Loading index  | model={model_name} | collection={collection}")
    index = load_index(model_name, collection)

    logger.info(f"Loading queries | model={model_name} | collection={collection}")
    query_enc = QueriesEncoding(model_name, collection)

    logger.info(f"Loading corpus mapping | model={model_name} | collection={collection}")
    corp_enc  = CorpusEncoding(model_name, collection)

    # offset → did mapper — needed to convert faiss int ids back to doc ids
    offset_to_did = corp_enc.get_ids()   # list where index = offset

    # ── search ─────────────────────────────────────────────────────────────────
    query_ids = query_enc.get_ids()
    qembs     = query_enc.get_encoding(query_ids)   # [N_queries, D]

    logger.info(f"Searching | n_queries={len(query_ids)} | k={k}")
    scores, offsets = index.search(qembs, k)        # both [N_queries, k]

    # ── build TREC run dataframe ───────────────────────────────────────────────
    out = []
    for i, qid in enumerate(query_ids):
        run = pd.DataFrame({
            "query_id": qid,
            "Q0":       "Q0",
            "doc_id":   [offset_to_did[o] for o in offsets[i]],
            "rank":     np.arange(k),
            "score":    scores[i],
            "run":      model_name,
        })
        out.append(run)

    out = pd.concat(out, ignore_index=True)
    logger.info(f"Search complete | {len(out)} results")
    return out


def save_run(run: pd.DataFrame, model_name: str, collection: str):
    """Save run to data/runs/{collection}/{model_name}.tsv in TREC format."""
    out_dir  = RUNS_DIR / collection
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model_name}.tsv"
    run.to_csv(out_path, sep="\t", header=None, index=False)
    logger.info(f"Run saved to {out_path}")
    return out_path