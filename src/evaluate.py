# src/evaluate.py

import logging
from pathlib import Path

import ir_measures
import pandas as pd

from src.config import RUNS_DIR, DATA_DIR
from src.data_loading import CollectionLoader

logger = logging.getLogger(__name__)

# ── Default measures ───────────────────────────────────────────────────────────

DEFAULT_MEASURES = ["nDCG@10", "AP", "R@1000", "RR@10"]


# ── Core evaluation ────────────────────────────────────────────────────────────

def evaluate(
    run: pd.DataFrame,
    qrels: pd.DataFrame,
    measures: list[str] = DEFAULT_MEASURES,
) -> pd.DataFrame:
    """
    Evaluate a run against qrels using ir_measures.

    Args:
        run:      TREC-format run — columns: query_id, Q0, doc_id, rank, score, run
        qrels:    relevance judgments — columns: query_id, doc_id, relevance
        measures: list of measure strings e.g. ["nDCG@10", "AP"]

    Returns:
        DataFrame with columns: query_id, measure, value
    """
    parsed   = [ir_measures.parse_measure(m) for m in measures]
    results  = pd.DataFrame(ir_measures.iter_calc(parsed, qrels, run))
    results["measure"] = results["measure"].astype(str)
    return results[["query_id", "measure", "value"]]


def summary(results: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-query results into mean scores per measure.

    Returns:
        DataFrame with columns: measure, mean
    """
    return (
        results
        .groupby("measure")["value"]
        .mean()
        .reset_index()
        .rename(columns={"value": "mean"})
        .sort_values("mean", ascending=False)
    )


# ── Load run from disk ─────────────────────────────────────────────────────────

def load_run(model_name: str, collection: str) -> pd.DataFrame:
    """Load a saved TREC run from data/runs/{collection}/{model_name}.tsv"""
    run_path = RUNS_DIR / collection / f"{model_name}.tsv"
    if not run_path.exists():
        raise FileNotFoundError(
            f"No run found at {run_path}. "
            f"Run pipelines/search.py --collection {collection} --model {model_name} first."
        )
    run = pd.read_csv(
        run_path, sep="\t", header=None,
        names=["query_id", "Q0", "doc_id", "rank", "score", "run"],
        dtype={"query_id": str, "doc_id": str},
    )
    logger.info(f"Loaded run: {run_path} | {run.query_id.nunique()} queries | {len(run)} results")
    return run


# ── Save results ───────────────────────────────────────────────────────────────

def save_results(results: pd.DataFrame, model_name: str, collection: str):
    """Save per-query evaluation results to data/results/{collection}/{model_name}.csv"""
    out_dir  = DATA_DIR / "results" / collection
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model_name}.csv"
    results.to_csv(out_path, index=False)
    logger.info(f"Results saved to {out_path}")
    return out_path