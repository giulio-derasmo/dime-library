# encode.py

import json
import logging
from datetime import datetime
from pathlib import Path
from time import time

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.config import (
    MEMMAPS_DIR,
    MODEL_TO_HF,
    MODEL_EMBEDDING_SIZE,
    ENCODING_BATCH_SIZE,
    get_corpus_name,
)

logger = logging.getLogger(__name__)


# ── Model ──────────────────────────────────────────────────────────────────────

def _load_model(model_name: str) -> SentenceTransformer:
    if model_name not in MODEL_TO_HF:
        raise ValueError(f"Unknown model '{model_name}'. Available: {list(MODEL_TO_HF)}")
    logger.info(f"Loading model: {model_name} ({MODEL_TO_HF[model_name]})")
    return SentenceTransformer(
        MODEL_TO_HF[model_name],
        model_kwargs={"use_safetensors": True}
    )


# ── Metadata ───────────────────────────────────────────────────────────────────

def _write_metadata(out_dir: Path, model_name: str, emb_size: int, n_items: int,
                    elapsed: float, batch_size: int, extra: dict = None):
    meta = {
        # model info
        "model":                  model_name,
        "hf_model":               MODEL_TO_HF[model_name],
        # shape info — everything memmap_interface needs to load without guessing
        "embedding_size":         emb_size,
        "dtype":                  "float32",
        "shape":                  [n_items, emb_size],
        "n_items":                n_items,
        # encoding info — for reproducibility
        "batch_size":             batch_size,
        "normalization":          "normalize_text.normalize",
        # status — incomplete file detection
        "status":                 "complete",
        "encoding_time_seconds":  round(elapsed, 2),
        "created":                datetime.now().isoformat(),
        # extra fields (corpus name, collection, split, ir_dataset_id etc.)
        **(extra or {}),
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"Metadata saved to {out_dir / 'metadata.json'}")


def read_metadata(out_dir: Path) -> dict:
    """Read and return metadata from a memmap directory."""
    meta_path = out_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"No metadata.json found at {meta_path}")
    with open(meta_path) as f:
        return json.load(f)


# ── Corpus ─────────────────────────────────────────────────────────────────────

def encode_corpus(
    docs: pd.DataFrame,            # columns: "did", "text"
    model_name: str,
    collection: str,
    batch_size: int = ENCODING_BATCH_SIZE,
    overwrite: bool = False,
) -> Path:
    """
    Encodes a document corpus and saves to data/memmaps/{model}/corpora/{corpus}/:
      - corpus.dat          float32 memmap, shape [N, D]
      - corpus_mapping.csv  did -> offset
      - metadata.json

    The corpus name is resolved automatically from the collection name via config.
    Encoding is skipped if corpus.dat already exists, unless overwrite=True.

    Returns the output directory path.
    """
    corpus_name = get_corpus_name(collection)
    out_dir  = MEMMAPS_DIR / model_name / "corpora" / corpus_name
    dat_path = out_dir / "corpus.dat"

    if dat_path.exists() and not overwrite:
        logger.info(f"Corpus memmap already exists at {dat_path}. Use overwrite=True to re-encode.")
        return out_dir

    out_dir.mkdir(parents=True, exist_ok=True)

    # sort by did for deterministic, stable offsets
    docs = docs.sort_values("did").reset_index(drop=True)
    docs["offset"] = np.arange(len(docs))

    model    = _load_model(model_name)
    emb_size = MODEL_EMBEDDING_SIZE[model_name]
    n_docs   = len(docs)
    texts    = docs["text"].tolist()

    logger.info(f"Encoding {n_docs} documents | model={model_name} | dim={emb_size} | corpus={corpus_name}")

    start = time()
    fp = np.memmap(dat_path, dtype="float32", mode="w+", shape=(n_docs, emb_size))
    for s in tqdm(range(0, n_docs, batch_size), desc=f"Encoding corpus ({corpus_name})"):
        e = min(s + batch_size, n_docs)
        fp[s:e] = model.encode(texts[s:e], show_progress_bar=False)
    fp.flush()
    elapsed = time() - start

    docs[["did", "offset"]].to_csv(out_dir / "corpus_mapping.csv", index=False)
    _write_metadata(
        out_dir, model_name, emb_size, n_docs, elapsed, batch_size,
        extra={
            "corpus":        corpus_name,
            "collection":    collection,
            "ir_dataset_id": f"corpus:{corpus_name}",
        }
    )

    logger.info(f"Corpus encoding done in {elapsed:.1f}s — saved to {out_dir}")
    return out_dir


# ── Queries ────────────────────────────────────────────────────────────────────

def encode_queries(
    queries: pd.DataFrame,         # columns: "qid", "text"
    model_name: str,
    collection: str,
    split_name: str = "queries",
    batch_size: int = ENCODING_BATCH_SIZE,
    overwrite: bool = False,
) -> Path:
    """
    Encodes queries and saves to data/memmaps/{model}/{collection}/:
      - {split_name}.dat          float32 memmap, shape [N, D]
      - {split_name}_mapping.tsv  qid -> offset
      - metadata.json

    Encoding is skipped if the .dat already exists, unless overwrite=True.

    Returns the output directory path.
    """
    out_dir  = MEMMAPS_DIR / model_name / collection
    dat_path = out_dir / f"{split_name}.dat"

    if dat_path.exists() and not overwrite:
        logger.info(f"Query memmap already exists at {dat_path}. Use overwrite=True to re-encode.")
        return out_dir

    out_dir.mkdir(parents=True, exist_ok=True)

    queries   = queries.reset_index(drop=True)
    queries["offset"] = np.arange(len(queries))

    model     = _load_model(model_name)
    emb_size  = MODEL_EMBEDDING_SIZE[model_name]
    n_queries = len(queries)
    texts     = queries["text"].tolist()

    logger.info(f"Encoding {n_queries} queries | model={model_name} | collection={collection}")

    start = time()
    # queries are small — encode in one shot with progress bar
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True)

    fp = np.memmap(dat_path, dtype="float32", mode="w+", shape=(n_queries, emb_size))
    fp[:] = embeddings
    fp.flush()
    elapsed = time() - start

    queries[["qid", "offset"]].to_csv(
        out_dir / f"{split_name}_mapping.tsv", sep="\t", index=False
    )
    _write_metadata(
        out_dir, model_name, emb_size, n_queries, elapsed, batch_size,
        extra={
            "collection": collection,
            "split":      split_name,
        }
    )

    logger.info(f"Query encoding done in {elapsed:.1f}s — saved to {out_dir}")
    return out_dir