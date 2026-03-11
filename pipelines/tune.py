# pipelines/tune.py
#
# Hyperparameter tuning for DIME filters using Optuna (TPE — Bayesian optimization).
#
# Mirrors the structure of dime.py exactly:
#   build_filter(args, filter_cfg, qrys_encoder)  — same signature, filter_cfg from trial
#   build_selector(selector_name, qrys_encoder, query_ids)  — identical to dime.py
#   main() calls both, then branches on rdime vs top-alpha for search
#
# ── Data roles ────────────────────────────────────────────────────────────────
#
#   dictCE  →  qrels only  (CE scores = graded relevance ground truth)
#   FAISS   →  run         (each model's own first-stage retrieval,
#                           loaded lazily inside build_filter only when needed)
#
# ── Workflow ──────────────────────────────────────────────────────────────────
#
#   Step 1 — build eval set (once, model-agnostic):
#     python pipelines/tune.py --build_eval_set
#     Saves:  data/tune/eval_queries.tsv   (query_id + text)
#             data/tune/eval_qrels.csv     (graded qrels from CE scores)
#
#   Step 2 — encode + retrieve (once per model):
#     python pipelines/tune.py --build_run -m contriever
#     Encodes eval queries into a dedicated 'tune_queries' memmap split,
#     then runs FAISS retrieval and saves:
#             data/tune/runs/contriever.tsv
#
#   Step 3 — run Optuna hyperparameter search:
#     python pipelines/tune.py -m contriever -f prf --n_trials 50
#     python pipelines/tune.py -m contriever -f prf-eclipse --n_trials 50
#     python pipelines/tune.py -m contriever -f gpt --n_trials 30
#
# ── Outputs ───────────────────────────────────────────────────────────────────
#
#   data/tune/configs/{model}__{filter}.yaml   ← best params, ready for dime.py --config
#   data/tune/{model}__{filter}__study.csv     ← full per-trial results

import argparse
import logging
import random

import numpy as np
import optuna
import pandas as pd
import yaml
from dotenv import load_dotenv

load_dotenv()

import sys
sys.path.append('/hdd4/giuder/Documents/projects/dime')

from src.config import MODEL_TO_HF, DATA_DIR, FILTERS, SELECTORS
from src.dime.masked_search import MaskedSearcher, SweepResults
from src.dime.selectors import TopAlphaSelector, RDIMESelector
from src.encode import encode_queries
from src.index import load_index
from src.memmap_interface import CorpusEncoding, CorpusMapping, QueriesEncoding

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ── Paths & constants ──────────────────────────────────────────────────────────

DICT_CE_PATH     = DATA_DIR / "other_files" / "dictCE.tsv"
TUNE_DIR         = DATA_DIR / "tune"
EVAL_QUERIES     = TUNE_DIR / "eval_queries.tsv"
EVAL_QRELS       = TUNE_DIR / "eval_qrels.csv"
TUNE_CONFIGS_DIR = TUNE_DIR / "configs"

N_EVAL_QUERIES  = 500
RANDOM_SEED     = 42
MEASURE         = "nDCG@100"

# MS MARCO passage corpus is shared across dl19/dl20/dlhard — use dl19 as proxy
# so we can reuse the existing corpus memmap and FAISS index.
EVAL_COLLECTION = "dl19"

# Dedicated split name for eval query memmaps — never overwrites dl19/dl20 queries.
EVAL_SPLIT      = "tune_queries"


def _eval_run_path(model_name: str):
    return TUNE_DIR / "runs" / f"{model_name}.tsv"


# ── Step 1: build eval set (model-agnostic, run once) ─────────────────────────

def build_eval_set(overwrite: bool = False):
    """
    Sample N_EVAL_QUERIES from dictCE, fetch their texts from ir_datasets,
    and save:
      - eval_queries.tsv : query_id + text  (text needed for encoding in Step 2)
      - eval_qrels.csv   : graded relevance from CE scores (4 levels, 0–3)

    CE scores are discretized into integer grades so ir_measures can compute
    nDCG with graded rather than binary relevance — gives a smoother tuning signal.
    """
    if not overwrite and EVAL_QUERIES.exists() and EVAL_QRELS.exists():
        logger.info("Eval set already exists — skipping. Use --overwrite to rebuild.")
        return

    TUNE_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading dictCE from {DICT_CE_PATH} ...")
    dictce = pd.read_csv(
        DICT_CE_PATH, sep="\t", header=None,
        names=["query_id", "na", "doc_id", "ce_score"],
        dtype={"query_id": str, "doc_id": str},
    ).drop(columns=["na"])

    all_qids = dictce["query_id"].unique()
    logger.info(f"Total queries in dictCE: {len(all_qids)}")

    rng         = random.Random(RANDOM_SEED)
    sampled     = sorted(rng.sample(list(all_qids), N_EVAL_QUERIES))
    sampled_set = set(sampled)
    eval_ce     = dictce[dictce["query_id"].isin(sampled_set)].copy()
    logger.info(f"Sampled {len(sampled)} queries | {len(eval_ce)} query-doc pairs")

    # ── fetch query texts from ir_datasets ────────────────────────────────────
    import ir_datasets
    from src.normalize_text import normalize
    logger.info("Fetching query texts from ir_datasets (msmarco-passage/train) ...")
    dataset     = ir_datasets.load("msmarco-passage/train")
    query_texts = {
        q.query_id: normalize(q.text)
        for q in dataset.queries_iter()
        if q.query_id in sampled_set
    }
    missing = sampled_set - set(query_texts.keys())
    if missing:
        logger.warning(f"{len(missing)} queries not found in ir_datasets — dropping.")
        sampled = [q for q in sampled if q in query_texts]

    queries_df = pd.DataFrame([
        {"query_id": qid, "text": query_texts[qid]}
        for qid in sampled
    ])
    queries_df.to_csv(EVAL_QUERIES, sep="\t", index=False)
    logger.info(f"Saved eval queries → {EVAL_QUERIES}  ({len(queries_df)} rows)")

    # ── graded qrels from CE scores ────────────────────────────────────────────
    bins   = [-np.inf, 0.0, 0.25, 0.75, np.inf]
    labels = [0, 1, 2, 3]
    eval_ce["relevance"] = pd.cut(eval_ce["ce_score"], bins=bins, labels=labels).astype(int)
    eval_ce[["query_id", "doc_id", "relevance"]].to_csv(EVAL_QRELS, index=False)
    logger.info(f"Saved eval qrels   → {EVAL_QRELS}")
    logger.info("── Eval set built ──")


# ── Step 2: encode eval queries + build model run (once per model) ─────────────

def build_model_run(model_name: str, k: int = 1000, overwrite: bool = False):
    """
    Encode the eval queries with the given model (stored as EVAL_SPLIT so they
    don't overwrite the dl19/dl20 memmaps), then run FAISS retrieval and save
    a TREC-format run to data/tune/runs/{model}.tsv.

    This is the run PRF and Eclipse will use — not the CE scores (those are qrels).
    Only needed for filters that read a retrieval run.
    """
    run_path = _eval_run_path(model_name)

    if not overwrite and run_path.exists():
        logger.info(f"Eval run already exists at {run_path} — skipping.")
        return

    if not EVAL_QUERIES.exists():
        raise FileNotFoundError(f"Run --build_eval_set first. Missing: {EVAL_QUERIES}")

    run_path.parent.mkdir(parents=True, exist_ok=True)

    # eval_queries.tsv has columns: query_id, text
    # encode_queries expects: qid, text — rename accordingly
    queries_df         = pd.read_csv(EVAL_QUERIES, sep="\t", dtype={"query_id": str})
    queries_for_encode = queries_df.rename(columns={"query_id": "qid"})

    # ── encode into dedicated split (never overwrites dl19/dl20 memmaps) ──────
    logger.info(f"Encoding eval queries | model={model_name} | split={EVAL_SPLIT}")
    encode_queries(
        queries=queries_for_encode,
        model_name=model_name,
        collection=EVAL_COLLECTION,
        split_name=EVAL_SPLIT,
        overwrite=overwrite,
    )

    # ── FAISS retrieval ────────────────────────────────────────────────────────
    logger.info(f"Loading index | model={model_name}")
    index          = load_index(model_name, EVAL_COLLECTION)
    qrys_encoder   = QueriesEncoding(model_name, EVAL_COLLECTION, split_name=EVAL_SPLIT)
    corpus_mapping = CorpusMapping(model_name, EVAL_COLLECTION)
    offset_to_did  = corpus_mapping.get_ids()

    eval_query_ids  = queries_df["query_id"].tolist()
    qembs           = qrys_encoder.get_encoding(eval_query_ids)    # [500, D]

    logger.info(f"Running FAISS search | n_queries={len(eval_query_ids)} | k={k}")
    scores, offsets = index.search(qembs, k)                       # [500, k]

    n_queries = len(eval_query_ids)
    run = pd.DataFrame({
        "query_id": np.repeat(eval_query_ids, k),
        "Q0":       "Q0",
        "doc_id":   [offset_to_did[o] for o in offsets.ravel()],
        "rank":     np.tile(np.arange(k), n_queries),
        "score":    scores.ravel(),
        "run":      model_name,
    })
    run.to_csv(run_path, sep="\t", header=False, index=False)
    logger.info(f"Saved eval run → {run_path}  ({len(run)} rows)")


# ── Load helper ────────────────────────────────────────────────────────────────

def _load_model_run(model_name: str) -> pd.DataFrame:
    run_path = _eval_run_path(model_name)
    if not run_path.exists():
        raise FileNotFoundError(
            f"Eval run not found at {run_path}. "
            f"Run: python pipelines/tune.py --build_run -m {model_name}"
        )
    return pd.read_csv(
        run_path, sep="\t", header=None,
        names=["query_id", "Q0", "doc_id", "rank", "score", "run"],
        dtype={"query_id": str, "doc_id": str},
    )


# ── build_filter — mirrors dime.py, filter_cfg comes from Optuna trial ────────

def build_filter(args, filter_cfg, qrys_encoder):
    """
    Identical structure to build_filter() in dime.py.
    filter_cfg is a plain dict — in dime.py it comes from a YAML file,
    here it is populated by the Optuna trial before this function is called.
    """
    def _load_docs_and_run():
        return CorpusEncoding(args.model, EVAL_COLLECTION), _load_model_run(args.model)

    eclipse      = args.filter.endswith("-eclipse")
    base_name    = args.filter.removesuffix("-eclipse")
    docs_encoder = None
    run          = None

    if base_name == "prf":
        from src.dime.filters.prf import PRFFilter
        docs_encoder, run = _load_docs_and_run()
        base_filter = PRFFilter(
            qrys_encoder=qrys_encoder,
            docs_encoder=docs_encoder,
            run=run,
            k=filter_cfg["k"],
        )

    elif base_name == "gpt":
        from src.dime.filters.gpt import GPTFilter
        base_filter = GPTFilter(
            qrys_encoder=qrys_encoder,
            model_name=args.model,
            collection=EVAL_COLLECTION,
            variant=filter_cfg["variant"],
        )

    elif base_name == "oracular":
        raise NotImplementedError("OracularFilter not yet implemented.")

    else:
        raise ValueError(f"Unknown filter: {args.filter!r}")

    if eclipse:
        from src.dime.filters.eclipse import EclipseWrapper
        if docs_encoder is None or run is None:
            docs_encoder, run = _load_docs_and_run()
        return EclipseWrapper(
            base_filter=base_filter,
            docs_encoder=docs_encoder,
            run=run,
            kneg=filter_cfg["kneg"],
            lambda_pos=filter_cfg.get("lambda_pos", 1.0),
            lambda_neg=filter_cfg.get("lambda_neg", 0.5),
        )

    return base_filter


# ── build_selector — identical to dime.py ─────────────────────────────────────

def build_selector(selector_name: str, qrys_encoder: QueriesEncoding, query_ids: list[str]):
    """Identical to build_selector() in dime.py."""
    if selector_name == "top-alpha":
        return TopAlphaSelector()

    if selector_name == "rdime":
        qembs = qrys_encoder.get_encoding(query_ids)
        return RDIMESelector(qembs)

    raise ValueError(f"Unknown selector: {selector_name!r}. Available: {SELECTORS}")


# ── Optuna objective ───────────────────────────────────────────────────────────

def make_objective(args, queries, qrels, qrys_encoder, corpus_mapping):
    """
    Closure over shared resources (index, encoders) loaded once and reused
    across all Optuna trials.

    The objective mirrors main() in dime.py:
      1. trial populates filter_cfg
      2. build_filter(args, filter_cfg, qrys_encoder)
      3. build_selector(args.selector, ...)
      4. branch on rdime vs top-alpha for search — same as dime.py
    """
    index     = load_index(args.model, EVAL_COLLECTION)
    query_ids = queries["query_id"].tolist()

    def objective(trial) -> float:

        # ── trial populates filter_cfg — same keys dime.py reads from YAML ────
        if args.filter.removesuffix("-eclipse") == "prf":
            filter_cfg = {"k": trial.suggest_int("k", 2, 20)}
        elif args.filter.removesuffix("-eclipse") == "gpt":
            filter_cfg = {"variant": "chatgpt4"}   # not a hyperparameter
        else:
            filter_cfg = {}

        if args.filter.endswith("-eclipse"):
            filter_cfg.update({
                "kneg":       trial.suggest_int("kneg", 2, 20),
                "lambda_pos": trial.suggest_float("lambda_pos", 0.5, 2.0),
                "lambda_neg": trial.suggest_float("lambda_neg", 0.1, 1.0),
            })

        # ── build filter + selector — identical calls to dime.py ───────────────
        dim_filter = build_filter(args, filter_cfg, qrys_encoder)
        selector   = build_selector(args.selector, qrys_encoder, query_ids)

        logger.info(f"Trial {trial.number:4d} | {dim_filter}")
        logger.info(f"Trial {trial.number:4d} | {selector}")

        # ── compute importance ─────────────────────────────────────────────────
        importance = dim_filter.compute(queries)

        # ── build searcher ─────────────────────────────────────────────────────
        searcher = MaskedSearcher(
            index=index,
            qrys_encoder=qrys_encoder,
            corpus_mapping=corpus_mapping,
            model_name=args.model,
            collection=EVAL_COLLECTION,
            selector=selector,
        )

        # ── rdime branch: single shot ──────────────────────────────────────────
        if args.selector == "rdime":
            results = searcher.run_once(
                dim_filter=dim_filter,
                importance=importance,
                k=100,
                save=False,
            )

        # ── top-alpha branch: alpha is also a trial parameter ──────────────────
        else:
            alpha   = trial.suggest_float("alpha", 0.1, 1.0, step=0.1)
            run     = searcher.search(importance, alpha=alpha, k=100)
            results = SweepResults(
                model_name=args.model,
                collection=EVAL_COLLECTION,
                filter_tag=dim_filter.tag,
                selector_tag=selector.tag,
                data={alpha: run},
            )

        # ── evaluate ───────────────────────────────────────────────────────────
        results.evaluate(qrels, measures=[MEASURE])
        mean_score = results.summary_table()[MEASURE].mean()

        logger.info(
            f"Trial {trial.number:4d} | {dim_filter.tag} | "
            f"{MEASURE}={mean_score:.4f} | params={trial.params}"
        )
        return mean_score

    return objective


# ── Save study + best config ───────────────────────────────────────────────────

def save_study(study, args):
    """
    Save two outputs:
      1. Full per-trial CSV          — for analysis / plotting
      2. Best params YAML in configs — ready to pass to dime.py --config
    """
    TUNE_DIR.mkdir(parents=True, exist_ok=True)
    TUNE_CONFIGS_DIR.mkdir(parents=True, exist_ok=True)

    stem = f"{args.model}__{args.filter}"

    study_path = TUNE_DIR / f"{stem}__study.csv"
    study.trials_dataframe().to_csv(study_path, index=False)
    logger.info(f"Study results → {study_path}")

    # best config — directly usable as --config in dime.py
    config_path = TUNE_CONFIGS_DIR / f"{stem}.yaml"
    with open(config_path, "w") as f:
        yaml.dump(study.best_params, f, default_flow_style=False)
    logger.info(f"Best config   → {config_path}  (pass to dime.py --config)")

    return study_path, config_path


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Hyperparameter tuning for DIME filters using Optuna.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--build_eval_set", action="store_true",
                        help="Step 1: sample eval queries + qrels from dictCE.")
    parser.add_argument("--build_run",      action="store_true",
                        help="Step 2: encode eval queries + FAISS retrieval for --model.")
    parser.add_argument("--overwrite",      action="store_true")
    parser.add_argument("-m", "--model",    choices=list(MODEL_TO_HF), default="contriever")
    parser.add_argument("-f", "--filter",   choices=FILTERS)
    parser.add_argument("-s", "--selector", choices=SELECTORS, default="top-alpha")
    parser.add_argument("--n_trials",       type=int, default=50)
    parser.add_argument("--n_jobs",         type=int, default=1,
                        help="Parallel Optuna workers (-1 = all cores).")
    parser.add_argument("--sampler",        choices=["tpe", "random"], default="tpe",
                        help="tpe = Bayesian (recommended), random = baseline.")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.build_eval_set:
        build_eval_set(overwrite=args.overwrite)
        return

    if args.build_run:
        build_model_run(args.model, overwrite=args.overwrite)
        return

    if not args.filter:
        raise ValueError("--filter is required for tuning.")

    queries   = pd.read_csv(EVAL_QUERIES, sep="\t", dtype={"query_id": str})
    qrels     = pd.read_csv(EVAL_QRELS,   dtype={"query_id": str, "doc_id": str})
    logger.info(f"Eval set: {len(queries)} queries | {len(qrels)} qrels")

    qrys_encoder   = QueriesEncoding(args.model, EVAL_COLLECTION, split_name=EVAL_SPLIT)
    corpus_mapping = CorpusMapping(args.model, EVAL_COLLECTION)

    sampler = (
        optuna.samplers.TPESampler(seed=RANDOM_SEED)
        if args.sampler == "tpe"
        else optuna.samplers.RandomSampler(seed=RANDOM_SEED)
    )
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name=f"{args.model}__{args.filter}",
    )

    objective = make_objective(args, queries, qrels, qrys_encoder, corpus_mapping)

    logger.info(f"── Optuna | filter={args.filter} | selector={args.selector} | trials={args.n_trials} ──")
    study.optimize(objective, n_trials=args.n_trials, n_jobs=args.n_jobs)

    print(f"\n── Best {MEASURE}: {study.best_value:.4f} ──")
    print(f"── Best params:     {study.best_params} ──\n")

    save_study(study, args)


if __name__ == "__main__":
    main()