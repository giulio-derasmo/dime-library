# repro_msmarco.py
#
# Micro reproducibility example using real MS MARCO data.
# Uses the actual project modules: CollectionLoader, encode_corpus,
# encode_queries, CorpusEncoding, QueriesEncoding.
#
# Run with:
#   python repro_msmarco.py
from dotenv import load_dotenv
load_dotenv()

import sys
sys.path.append('/hdd4/giuder/Documents/projects/dime')

import numpy as np

from src.data_loading import CollectionLoader
from src.encode import encode_corpus, encode_queries, _load_model
from src.memmap_interface import CorpusEncoding, QueriesEncoding
from src.config import MODEL_TO_HF

# ── Config ─────────────────────────────────────────────────────────────────────

COLLECTION = "DL19"
MODEL_NAME = "contriever"
N_DOCS     = 1000   # subset size — increase for a larger test
N_QUERIES  = 50     # dl19 has 43 queries total


# ── Step 1: Load a small subset ────────────────────────────────────────────────

def load_subset():
    print("\n" + "="*60)
    print("STEP 1 — Loading subset via CollectionLoader")
    print("="*60)

    loader  = CollectionLoader(COLLECTION)
    print(loader)

    docs    = loader.corpus().head(N_DOCS)
    queries = loader.queries().head(N_QUERIES)
    qrels   = loader.qrels()

    print(f"\nDocs    ({len(docs)}):\n{docs.head(3).to_string(index=False)}")
    print(f"\nQueries ({len(queries)}):\n{queries.to_string(index=False)}")
    print(f"\nQrels   ({len(qrels)}): {qrels.query_id.nunique()} queries with relevance judgments")

    return docs, queries


# ── Step 2: Encode and store ───────────────────────────────────────────────────

def encode(docs, queries):
    print("\n" + "="*60)
    print("STEP 2 — Encoding via encode_corpus / encode_queries")
    print("="*60)

    corpus_dir = encode_corpus(
        docs,
        model_name=MODEL_NAME,
        collection=COLLECTION,
        overwrite=True,
    )
    query_dir = encode_queries(
        queries,
        model_name=MODEL_NAME,
        collection=COLLECTION,
        overwrite=True,
    )

    print(f"\nCorpus stored at:  {corpus_dir}")
    print(f"Queries stored at: {query_dir}")


# ── Step 3: Load back via interface ───────────────────────────────────────────

def load():
    print("\n" + "="*60)
    print("STEP 3 — Loading back via CorpusEncoding / QueriesEncoding")
    print("="*60)

    corp_enc  = CorpusEncoding(MODEL_NAME, COLLECTION)
    query_enc = QueriesEncoding(MODEL_NAME, COLLECTION)

    print(f"\n{corp_enc}")
    print(f"{query_enc}")

    return corp_enc, query_enc


# ── Step 4: Verify ─────────────────────────────────────────────────────────────

def verify(docs, queries, corp_enc, query_enc):
    print("\n" + "="*60)
    print("STEP 4 — Verifying id mapping and embedding correctness")
    print("="*60)

    from sentence_transformers import SentenceTransformer
    model = _load_model(MODEL_NAME)

    all_passed = True

    # --- corpus ---
    # encode() sorts docs by did before storing — mirror that here
    print("\n[Corpus — id → embedding correctness]")
    docs_sorted = docs.sort_values("did").reset_index(drop=True)
    for _, row in docs_sorted.iterrows():
        did          = row["did"]
        loaded_emb   = corp_enc.get_encoding(did)
        original_emb = model.encode(row["text"], show_progress_bar=False)
        match        = np.allclose(loaded_emb, original_emb, atol=1e-5)
        status       = "✓ PASS" if match else "✗ FAIL"
        print(f"  {status} | did={did} | emb[:3]={loaded_emb[:3].round(4)}")
        if not match:
            all_passed = False

    # --- queries ---
    print("\n[Queries — id → embedding correctness]")
    for _, row in queries.iterrows():
        qid          = row["qid"]
        loaded_emb   = query_enc.get_encoding(qid)
        original_emb = model.encode(row["text"], show_progress_bar=False)
        match        = np.allclose(loaded_emb, original_emb, atol=1e-5)
        status       = "✓ PASS" if match else "✗ FAIL"
        print(f"  {status} | qid={qid} | emb[:3]={loaded_emb[:3].round(4)}")
        if not match:
            all_passed = False

    # --- retrieval sanity ---
    print("\n[Retrieval — query → top-1 document (dot product over corpus subset)]")
    corpus_matrix = corp_enc.get_all()
    doc_ids       = corp_enc.get_ids()

    for _, row in queries.iterrows():
        qid    = row["qid"]
        qemb   = query_enc.get_encoding(qid)
        scores = corpus_matrix @ qemb
        top1   = doc_ids[int(np.argmax(scores))]
        print(f"  qid={qid} | '{row['text'][:50]}'")
        print(f"           → top1 did={top1} | score={scores.max():.4f}")

    print("\n" + "="*60)
    if all_passed:
        print("ALL ID/EMBEDDING CHECKS PASSED ✓")
    else:
        print("SOME CHECKS FAILED ✗ — see above")
    print("="*60)


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    docs, queries       = load_subset()
    encode(docs, queries)
    corp_enc, query_enc = load()
    verify(docs, queries, corp_enc, query_enc)