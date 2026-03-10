# memmap_interface.py

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize as sk_normalize

from src.config import MEMMAPS_DIR, get_corpus_name

logger = logging.getLogger(__name__)


# ── Base ───────────────────────────────────────────────────────────────────────

class MemmapEncoding:
    """
    Base class for reading memmap embeddings.

    - Loads embedding size, shape and dtype from metadata.json — nothing hardcoded
    - Validates integrity on load (status must be 'complete')
    - Fast O(1) id -> offset lookup via plain Python dict
    - Accepts both single id (str) and list of ids in get_encoding()
    """

    def __init__(self, dat_path: Path, mapping_path: Path, id_col: str, sep: str = ","):

        # ── load and validate metadata ─────────────────────────────────────────
        meta      = self._load_metadata(dat_path)
        emb_size  = meta["embedding_size"]
        dtype     = meta.get("dtype", "float32")
        n_items   = meta["n_items"]

        self.meta     = meta
        self.emb_size = emb_size

        # ── load memmap in read-only mode ──────────────────────────────────────
        self.data  = np.memmap(dat_path, dtype=dtype, mode="r").reshape(n_items, emb_size)
        self.shape = self.data.shape

        # ── build O(1) lookup dict ─────────────────────────────────────────────
        mapping           = pd.read_csv(mapping_path, dtype={id_col: str}, sep=sep)
        self.id_to_offset = dict(zip(mapping[id_col], mapping["offset"]))
        self.ids          = list(self.id_to_offset.keys())

        logger.info(f"Loaded {self.__class__.__name__}: shape={self.shape} | path={dat_path}")

    # ── metadata ───────────────────────────────────────────────────────────────

    @staticmethod
    def _load_metadata(dat_path: Path) -> dict:
        meta_path = dat_path.parent / "metadata.json"

        if not meta_path.exists():
            raise FileNotFoundError(
                f"No metadata.json found at {meta_path}. "
                f"Re-run encode_corpus() or encode_queries() to regenerate."
            )

        with open(meta_path) as f:
            meta = json.load(f)

        # integrity check — catches incomplete/corrupt encodings
        if meta.get("status") != "complete":
            raise RuntimeError(
                f"Memmap at {dat_path} has status='{meta.get('status')}'. "
                f"The encoding may be incomplete or corrupt. Re-run encoding with overwrite=True."
            )

        return meta

    def get_meta(self) -> dict:
        """Return the full metadata dict for inspection."""
        return self.meta

    # ── retrieval ──────────────────────────────────────────────────────────────

    def get_encoding(self, ids: str | list[str]) -> np.ndarray:
        """
        Retrieve embeddings for one or more ids.

        Args:
            ids: a single id string or a list of id strings
        Returns:
            np.ndarray of shape [D] for a single id, [N, D] for a list
        """
        if isinstance(ids, str):
            if ids not in self.id_to_offset:
                raise KeyError(f"Id '{ids}' not found in memmap.")
            return self.data[self.id_to_offset[ids]]

        missing = [i for i in ids if i not in self.id_to_offset]
        if missing:
            raise KeyError(f"{len(missing)} ids not found in memmap: {missing[:5]}{'...' if len(missing) > 5 else ''}")

        offsets = [self.id_to_offset[i] for i in ids]
        return self.data[offsets]

    def get_all(self, normalized: bool = False) -> np.ndarray:
        """
        Return the full embedding matrix.
        Materializes the memmap into RAM — use only when you need all vectors (e.g. FAISS indexing).
        """
        if normalized:
            return self._get_normalized()
        return np.array(self.data)

    def get_normalized_encoding(self, ids: str | list[str]) -> np.ndarray:
        """Retrieve L2-normalized embeddings for one or more ids."""
        normalized = self._get_normalized()
        if isinstance(ids, str):
            return normalized[self.id_to_offset[ids]]
        offsets = [self.id_to_offset[i] for i in ids]
        return normalized[offsets]

    def get_centroid(self) -> np.ndarray:
        """Return the mean vector of all embeddings. Cached after first call."""
        if not hasattr(self, "_centroid"):
            self._centroid = np.mean(self.data, axis=0)
        return self._centroid

    # ── helpers ────────────────────────────────────────────────────────────────

    def _get_normalized(self) -> np.ndarray:
        """L2-normalize the full matrix. Cached after first call."""
        if not hasattr(self, "_normalized_data"):
            self._normalized_data = sk_normalize(np.array(self.data))
        return self._normalized_data

    def get_ids(self) -> list[str]:
        return self.ids

    def __len__(self) -> int:
        return self.data.shape[0]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"n={len(self)}, "
            f"emb_size={self.emb_size}, "
            f"model={self.meta.get('model')}, "
            f"created={self.meta.get('created', 'unknown')})"
        )


# ── Corpus ─────────────────────────────────────────────────────────────────────

class CorpusEncoding(MemmapEncoding):
    """
    Memmap interface for a document corpus.
    Corpus path is resolved automatically from model + collection via config.

    Usage:
        enc  = CorpusEncoding("contriever", "dl19")
        embs = enc.get_encoding(["doc1", "doc2"])
        all  = enc.get_all()   # for FAISS indexing
    """

    def __init__(self, model_name: str, collection: str):
        corpus_name = get_corpus_name(collection)
        base        = MEMMAPS_DIR / model_name / "corpora" / corpus_name

        self._validate_dir(base)
        super().__init__(
            dat_path=base / "corpus.dat",
            mapping_path=base / "corpus_mapping.csv",
            id_col="did",
            sep=",",
        )
        self.model_name  = model_name
        self.corpus_name = corpus_name

    @staticmethod
    def _validate_dir(base: Path):
        if not base.exists():
            raise FileNotFoundError(
                f"Corpus memmap directory not found: {base}. "
                f"Run encode_corpus() first."
            )


# ── Queries ────────────────────────────────────────────────────────────────────

class QueriesEncoding(MemmapEncoding):
    """
    Memmap interface for a query set.
    Query path is resolved automatically from model + collection via config.

    Usage:
        enc  = QueriesEncoding("contriever", "dl19")
        embs = enc.get_encoding(["q1", "q2"])
    """

    def __init__(self, model_name: str, collection: str, split_name: str = "queries"):
        base = MEMMAPS_DIR / model_name / collection

        self._validate_dir(base, collection)
        super().__init__(
            dat_path=base / f"{split_name}.dat",
            mapping_path=base / f"{split_name}_mapping.tsv",
            id_col="qid",
            sep="\t",
        )
        self.model_name  = model_name
        self.collection  = collection
        self.split_name  = split_name

    @staticmethod
    def _validate_dir(base: Path, collection: str):
        if not base.exists():
            raise FileNotFoundError(
                f"Query memmap directory not found: {base}. "
                f"Run encode_queries() for collection='{collection}' first."
            )