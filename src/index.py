# index.py

import logging
from pathlib import Path
from time import time

import faiss
import numpy as np
from tqdm import tqdm

from src.config import INDICES_DIR, get_corpus_name
from src.memmap_interface import CorpusEncoding

logger = logging.getLogger(__name__)


def build_index(
    model_name: str,
    collection: str,
    overwrite: bool = False,
) -> Path:
    """
    Builds a FAISS flat inner product index from a corpus memmap and saves it.

    Requires encode_corpus() to have been run first.
    Returns the path to the saved index.
    """
    corpus_name = get_corpus_name(collection)
    out_dir     = INDICES_DIR / model_name / corpus_name
    index_path  = out_dir / "index.faiss"

    if index_path.exists() and not overwrite:
        logger.info(f"Index already exists at {index_path}. Use overwrite=True to rebuild.")
        return index_path

    out_dir.mkdir(parents=True, exist_ok=True)

    # load corpus from memmap — no path management needed
    logger.info(f"Loading corpus | model={model_name} | corpus={corpus_name}")
    corp_enc = CorpusEncoding(model_name, collection)
    n_docs, emb_size = corp_enc.shape

    # build flat IP index — exact search, no approximation
    logger.info(f"Building FAISS IndexFlatIP | n_docs={n_docs} | emb_size={emb_size}")
    index = faiss.IndexFlatIP(emb_size)

    # add in batches — avoids materializing the full matrix at once
    start      = time()
    batch_size = 1024
    for s in tqdm(range(0, n_docs, batch_size), desc="Adding to FAISS index"):
        e = min(s + batch_size, n_docs)
        index.add(np.array(corp_enc.data[s:e]))  # slice memmap directly
    elapsed = time() - start

    faiss.write_index(index, str(index_path))

    logger.info(f"Index built in {elapsed:.1f}s — {index.ntotal} vectors | saved to {index_path}")
    return index_path


def load_index(model_name: str, collection: str) -> faiss.Index:
    """
    Loads a saved FAISS index from disk.

    Usage:
        index = load_index("contriever", "dl19")
        scores, ids = index.search(query_embs, k=1000)
    """
    corpus_name = get_corpus_name(collection)
    index_path  = INDICES_DIR / model_name / corpus_name / "index.faiss"

    if not index_path.exists():
        raise FileNotFoundError(
            f"No FAISS index found at {index_path}. "
            f"Run build_index('{model_name}', '{collection}') first."
        )

    logger.info(f"Loading FAISS index from {index_path}")
    return faiss.read_index(str(index_path))