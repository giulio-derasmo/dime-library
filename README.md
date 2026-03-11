# DIME — Dimension-Importance Masked Embeddings

DIME is a research framework for **dimension-selective dense retrieval**. Instead of searching with a full-dimensional query embedding, DIME computes a per-query importance score over every embedding dimension and masks (or weights) out the less informative ones before querying a FAISS index. This enables controlled experiments over how much of the embedding space is actually needed for effective retrieval.

---

## Table of Contents

- [Core Idea](#core-idea)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Pipelines](#pipelines)
  - [1. Build — encode and index](#1-build--encode-and-index)
  - [2. Search — baseline retrieval](#2-search--baseline-retrieval)
  - [3. DIME — masked retrieval sweep](#3-dime--masked-retrieval-sweep)
- [Key Concepts](#key-concepts)
  - [ImportanceMatrix](#importancematrix)
  - [DimeFilter](#dimefilter)
  - [DimSelector](#dimselector)
  - [MaskedSearcher and SweepResults](#maskedsearcher-and-sweepresults)
- [Data Layout](#data-layout)
- [Extending DIME](#extending-dime)
  - [Adding a new filter](#adding-a-new-filter)
  - [Adding a new selector](#adding-a-new-selector)
  - [Adding a new collection](#adding-a-new-collection)
  - [Adding a new model](#adding-a-new-model)
- [Using the Library API directly](#using-the-library-api-directly)
- [Evaluation Measures](#evaluation-measures)
- [Config Files](#config-files)

---

## Core Idea

Standard dense retrieval computes `score(q, d) = q · d` over all D dimensions of the embedding. DIME asks: *which dimensions matter for each query?*

The pipeline has three stages:

```
query + corpus embeddings
        │
        ▼
  DimeFilter.compute()        ← how important is each dimension?
        │
        ▼
  ImportanceMatrix [N, D]     ← per-query importance scores
        │
        ▼
  DimSelector.select(α)       ← turn scores into weights [N, D]
        │
        ▼
  masked_q = q ⊙ weights      ← apply weights element-wise
        │
        ▼
  FAISS index.search()        ← retrieve top-k documents
```

Sweeping `α ∈ {0.1, 0.2, ..., 1.0}` lets you measure the retrieval quality as a function of how many dimensions are kept.

---

## Project Structure

```
dime/
├── configs/                        # YAML configs for filters
│   └── prf_k10.yaml
│
├── data/                           # generated artifacts — gitignore this
│   ├── memmaps/                    # float32 embedding memmaps
│   │   └── {model}/
│   │       ├── corpora/{corpus}/   # corpus.dat, corpus_mapping.csv, metadata.json
│   │       └── {collection}/       # queries.dat, queries_mapping.tsv, metadata.json
│   ├── vectordb/                   # FAISS indices
│   │   └── {model}/{corpus}/index.faiss
│   ├── runs/                       # TREC-format run TSVs
│   │   └── {collection}/
│   │       ├── {model}.tsv                              # baseline run
│   │       └── {model}__{filter}__{selector}.tsv        # DIME sweep run
│   └── results/                    # per-query evaluation CSVs
│       └── {collection}/
│           ├── {model}.csv
│           └── {model}__{filter}__{selector}.csv
│
├── pipelines/                      # entry-point scripts
│   ├── build.py                    # encode corpus + queries, build index
│   ├── search.py                   # baseline full-dim retrieval
│   └── dime.py                     # DIME importance + masked sweep
│
└── src/
    ├── config.py                   # paths, models, collections, measures
    ├── data_loading.py             # CollectionLoader
    ├── normalize_text.py           # unicode normalization
    ├── encode.py                   # encode_corpus(), encode_queries()
    ├── index.py                    # build_index(), load_index()
    ├── memmap_interface.py         # CorpusEncoding, QueriesEncoding, CorpusMapping
    ├── search.py                   # baseline search()
    ├── evaluate.py                 # evaluate(), summary(), load_run(), save_results()
    └── dime/
        ├── importance.py           # ImportanceMatrix dataclass
        ├── filters/
        │   ├── base.py             # DimeFilter ABC
        │   ├── prf.py              # PRFFilter
        │   └── oracular.py        # OracularFilter
        └── selectors/
            ├── base.py             # DimSelector ABC
            ├── top_alpha.py        # TopAlphaSelector (default)
            └── your_method.py      # add new selectors here
```

---

## Configuration

All models, collections, paths and default measures live in a single file: `src/config.py`.

**Default models**

| Key | HuggingFace model |
|---|---|
| `tasb` | `sentence-transformers/msmarco-distilbert-base-tas-b` |
| `contriever` | `facebook/contriever-msmarco` |
| `cocondenser` | `sentence-transformers/msmarco-bert-co-condensor` |
| `ance` | `sentence-transformers/msmarco-roberta-base-ance-firstp` |

All models produce 768-dimensional embeddings.

**Default collections**

| Key | Corpus | ir_datasets id |
|---|---|---|
| `dl19` | msmarco-passage | `msmarco-passage/trec-dl-2019/judged` |
| `dl20` | msmarco-passage | `msmarco-passage/trec-dl-2020/judged` |
| `dlhard` | msmarco-passage | `msmarco-passage/trec-dl-hard` |

`dl19` and `dl20` share the same corpus (`msmarco-passage`), so corpus encoding only needs to run once and is reused across both collections.

**Default evaluation measures**: `nDCG@10`, `AP`, `R@1000`, `RR@10`

---

## Pipelines

The three pipelines must be run in order for a given `(collection, model)` pair.

### 1. Build — encode and index

Encodes the corpus and queries into float32 memmap files and builds a FAISS flat inner-product index.

```bash
python pipelines/build.py -c <collection> -m <model> [options]
```

**Arguments**

| Argument | Required | Default | Description |
|---|---|---|---|
| `-c`, `--collection` | yes | — | Collection to build (`dl19`, `dl20`, `dlhard`) |
| `-m`, `--model` | yes | — | Model to use (`contriever`, `tasb`, `ance`, `cocondenser`) |
| `--overwrite` | no | `False` | Re-encode and rebuild even if outputs already exist |
| `--skip_corpus` | no | `False` | Skip corpus encoding (useful when collections share a corpus) |
| `--skip_queries` | no | `False` | Skip query encoding |
| `--skip_index` | no | `False` | Skip FAISS index building |

**Examples**

```bash
# Full build from scratch
python pipelines/build.py -c dl19 -m contriever

# dl20 shares the same corpus as dl19 — skip re-encoding it
python pipelines/build.py -c dl20 -m contriever --skip_corpus --skip_index

# Force rebuild of everything
python pipelines/build.py -c dl19 -m contriever --overwrite

# Only rebuild the FAISS index (corpus and queries already encoded)
python pipelines/build.py -c dl19 -m contriever --skip_corpus --skip_queries
```

**Outputs**

```
data/memmaps/{model}/corpora/{corpus}/
    corpus.dat              # float32 memmap, shape [N_docs, D]
    corpus_mapping.csv      # did → offset
    metadata.json           # shape, model, encoding info

data/memmaps/{model}/{collection}/
    queries.dat             # float32 memmap, shape [N_queries, D]
    queries_mapping.tsv     # qid → offset
    metadata.json

data/vectordb/{model}/{corpus}/
    index.faiss             # FAISS IndexFlatIP
```

---

### 2. Search — baseline retrieval

Runs full-dimensional retrieval (no masking) and optionally evaluates the run.

```bash
python pipelines/search.py -c <collection> -m <model> [options]
```

**Arguments**

| Argument | Required | Default | Description |
|---|---|---|---|
| `-c`, `--collection` | yes | — | Collection to search |
| `-m`, `--model` | yes | — | Model to use |
| `-k`, `--k` | no | `1000` | Documents retrieved per query |
| `--evaluate` | no | `False` | Evaluate the run after searching |
| `--overwrite` | no | `False` | Re-run search even if run already exists on disk |
| `--measures` | no | `nDCG@10 AP R@1000 RR@10` | Evaluation measures to compute |

**Examples**

```bash
# Search only
python pipelines/search.py -c dl19 -m contriever

# Search and evaluate
python pipelines/search.py -c dl19 -m contriever --evaluate

# Evaluate with custom measures
python pipelines/search.py -c dl19 -m contriever --evaluate --measures nDCG@10 RR@10

# Force re-run even if run exists
python pipelines/search.py -c dl19 -m contriever --evaluate --overwrite
```

**Outputs**

```
data/runs/{collection}/{model}.tsv          # TREC-format run (tab-separated, no header)
data/results/{collection}/{model}.csv       # per-query metrics (if --evaluate)
```

The run TSV has columns: `query_id  Q0  doc_id  rank  score  run`

> **Note**: The baseline run is required before running the `prf` filter in the DIME pipeline — PRF needs a ranked list to identify pseudo-relevant documents.

---

### 3. DIME — masked retrieval sweep

Computes per-query dimension importance, sweeps over `α ∈ [0.1, 1.0]`, evaluates every alpha and saves results.

```bash
python pipelines/dime.py -c <collection> -m <model> -f <filter> --config <yaml> [options]
```

**Arguments**

| Argument | Required | Default | Description |
|---|---|---|---|
| `-c`, `--collection` | yes | — | Collection |
| `-m`, `--model` | yes | — | Model |
| `-f`, `--filter` | yes | — | Importance filter (`prf`, `oracular`) |
| `-s`, `--selector` | no | `top-alpha` | Dimension selection strategy |
| `--config` | yes | — | Path to filter config YAML |
| `--alphas` | no | `0.1 0.2 … 1.0` | Alpha values to sweep |
| `--k` | no | `1000` | Documents retrieved per query |
| `--n_jobs` | no | `1` | Parallel threads (`-1` = one per alpha) |
| `--measures` | no | default measures | Evaluation measures |
| `--save` | no | `False` | Save sweep runs to disk |
| `--overwrite` | no | `False` | Re-run even if sweep already exists |

**Examples**

```bash
# PRF filter, default alpha sweep, single-threaded
python pipelines/dime.py -c dl19 -m contriever -f prf --config configs/prf_k10.yaml

# PRF filter, parallel sweep across all alphas
python pipelines/dime.py -c dl19 -m contriever -f prf --config configs/prf_k10.yaml --n_jobs -1

# Oracular filter (upper bound — uses ground-truth qrels)
python pipelines/dime.py -c dl19 -m contriever -f oracular --config configs/oracular.yaml

# Custom alpha grid
python pipelines/dime.py -c dl19 -m contriever -f prf --config configs/prf_k10.yaml --alphas 0.1 0.3 0.5 1.0

# Use a different selector
python pipelines/dime.py -c dl19 -m contriever -f prf --config configs/prf_k10.yaml --selector your-method

# Save runs to disk and skip re-running if they exist
python pipelines/dime.py -c dl19 -m contriever -f prf --config configs/prf_k10.yaml --save
```

**Outputs**

```
data/runs/{collection}/{model}__{filter}__{selector}.tsv
    # Combined TSV for all alphas. Extra column `alpha` identifies each run.

data/results/{collection}/{model}__{filter}__{selector}.csv
    # Per-query metrics for every alpha.
    # Columns: alpha, query_id, measure, value
```


