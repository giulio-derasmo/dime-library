from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR    = ROOT_DIR / "data"
MEMMAPS_DIR  = DATA_DIR / "memmaps"
INDICES_DIR  = DATA_DIR / "vectordb"
RUNS_DIR = DATA_DIR / "runs"

# ── Models ─────────────────────────────────────────────────────────────────────
MODEL_TO_HF = {
    "tasb":        "sentence-transformers/msmarco-distilbert-base-tas-b",
    "contriever":  "facebook/contriever-msmarco",
    "cocondenser": "sentence-transformers/msmarco-bert-co-condensor",
    "ance":        "sentence-transformers/msmarco-roberta-base-ance-firstp",
}

MODEL_EMBEDDING_SIZE = {
    "tasb":        768,
    "contriever":  768,
    "cocondenser": 768,
    "ance":        768,
}

ENCODING_BATCH_SIZE = 2048

# ── Collections ────────────────────────────────────────────────────────────────
# Single source of truth — everything is derived from here
COLLECTIONS = {
    "dl19": {
        "corpus":          "msmarco-passage",
        "ir_dataset_docs": "msmarco-passage",
        "ir_dataset_queries_qrels": "msmarco-passage/trec-dl-2019/judged",
    },
    "dl20": {
        "corpus":          "msmarco-passage",
        "ir_dataset_docs": "msmarco-passage",
        "ir_dataset_queries_qrels": "msmarco-passage/trec-dl-2020/judged",
    },
}

def get_corpus_name(collection: str) -> str:
    return COLLECTIONS[collection]["corpus"]

def get_ir_dataset_docs(collection: str) -> str:
    return COLLECTIONS[collection]["ir_dataset_docs"]

def get_ir_dataset_queries_qrels(collection: str) -> str:
    return COLLECTIONS[collection]["ir_dataset_queries_qrels"]