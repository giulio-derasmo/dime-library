# src/search.py

import logging

import numpy as np
import pandas as pd

from src.config import RUNS_DIR, get_corpus_name
from src.index import load_index
from src.memmap_interface import CorpusMapping, QueriesEncoding

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
    logger.info(f"Loading index   | model={model_name} | collection={collection}")
    index = load_index(model_name, collection)

    logger.info(f"Loading queries | model={model_name} | collection={collection}")
    query_enc = QueriesEncoding(model_name, collection)

    # CorpusMapping: only loads corpus_mapping.csv — no memmap
    logger.info(f"Loading corpus mapping | model={model_name} | collection={collection}")
    corpus_mapping = CorpusMapping(model_name, collection)
    offset_to_did  = corpus_mapping.get_ids()   # list[str], index = faiss offset

    # ── search ─────────────────────────────────────────────────────────────────
    query_ids = query_enc.get_ids()
    qembs     = query_enc.get_encoding(query_ids)       # [N, D]

    logger.info(f"Searching | n_queries={len(query_ids)} | k={k}")
    scores, offsets = index.search(qembs, k)            # [N, k]

    # ── build TREC run dataframe ───────────────────────────────────────────────
    n_queries = len(query_ids)
    run = pd.DataFrame({
        "query_id": np.repeat(query_ids, k),
        "Q0":       "Q0",
        "doc_id":   [offset_to_did[o] for o in offsets.ravel()],
        "rank":     np.tile(np.arange(k), n_queries),
        "score":    scores.ravel(),
        "run":      model_name,
    })

    logger.info(f"Search complete | {len(run)} results")
    return run


def save_run(run: pd.DataFrame, model_name: str, collection: str):
    """Save run to data/runs/{collection}/{model_name}.tsv in TREC format."""
    out_dir  = RUNS_DIR / collection
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model_name}.tsv"
    run.to_csv(out_path, sep="\t", header=None, index=False)
    logger.info(f"Run saved to {out_path}")
    return out_path