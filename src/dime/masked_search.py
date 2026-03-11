# src/dime/masked_search.py
#
# MaskedSearcher: applies per-query dimension weights to query embeddings
# and searches a FAISS index. Supports single-alpha search and efficient
# alpha sweeps.
#
# Dimension selection is delegated to a DimSelector, which is injected
# at construction time. This decouples the masking strategy from the
# search logic — swap in any DimSelector subclass without touching this file.
#
# Parallelism: sweep() uses ThreadPoolExecutor over alphas.
# FAISS releases the GIL during index.search, so threads are safe and
# give near-linear speedup up to len(alphas) workers.
# ProcessPoolExecutor is intentionally avoided — it would require reloading
# the FAISS index and memmap encodings in every subprocess.

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Sequence

import faiss
import numpy as np
import pandas as pd

from src.config import RUNS_DIR, DEFAULT_MEASURES, DATA_DIR
from src.dime.filters.base import DimeFilter
from src.dime.selectors.base import DimSelector
from src.dime.selectors.top_alpha import TopAlphaSelector
from src.dime.importance import ImportanceMatrix
from src.evaluate import evaluate, summary
from src.memmap_interface import CorpusMapping, QueriesEncoding

logger = logging.getLogger(__name__)

DEFAULT_ALPHAS = np.round(np.arange(0.1, 1.1, 0.1), 2).tolist()


class MaskedSearcher:
    """
    Applies per-query dimension weights to query embeddings, then searches
    a FAISS index for the top-k documents.

    Dimension weighting is fully delegated to the injected DimSelector.
    The default selector is TopAlphaSelector (original DIME behavior).

    Args:
        index:          FAISS index (must support .search)
        qrys_encoder:   QueriesEncoding — provides raw query embeddings
        corpus_mapping: CorpusMapping   — lightweight offset → doc_id lookup
        model_name:     string tag used in filenames and run column
        collection:     collection name, used for save paths
        selector:       DimSelector to use for weighting (default: TopAlphaSelector)
    """

    def __init__(
        self,
        index: faiss.Index,
        qrys_encoder: QueriesEncoding,
        corpus_mapping: CorpusMapping,
        model_name: str,
        collection: str,
        selector: DimSelector | None = None,
    ):
        self.index          = index
        self.qrys_encoder   = qrys_encoder
        self.corpus_mapping = corpus_mapping
        self.model_name     = model_name
        self.collection     = collection
        self.selector       = selector or TopAlphaSelector()
        self._offset_to_did = corpus_mapping.get_ids()   # list[str], index = faiss offset

    # ── single alpha ───────────────────────────────────────────────────────────

    def search(
        self,
        importance: ImportanceMatrix,
        alpha: float,
        k: int = 1000,
    ) -> pd.DataFrame:
        """
        Search with query embeddings weighted by the selector for this alpha.

        Returns a TREC-format run DataFrame:
            query_id | Q0 | doc_id | rank | score | run | alpha
        """
        weights  = self.selector.select(importance, alpha)           # [N, D] float32
        qembs    = self.qrys_encoder.get_encoding(importance.query_ids)
        masked_q = (qembs * weights).astype(np.float32)
        scores, offsets = self.index.search(masked_q, k)
        return self._build_run(importance.query_ids, scores, offsets, k, alpha)

    # ── single shot — for alpha-free selectors (e.g. RDIME) ──────────────────

    def run_once(
        self,
        dim_filter: DimeFilter,
        importance: ImportanceMatrix,
        k: int = 1000,
        save: bool = False,
    ) -> "SweepResults":
        """
        Run a single masked search without an alpha grid.

        Intended for selectors that determine their own threshold internally
        (e.g. RDIMESelector). The result is stored in a SweepResults container
        under the sentinel key "rdime", which round-trips cleanly through
        CSV save/load (unlike None, which becomes NaN on reload).

        The full persistence and evaluation API (.evaluate(), .save_results(),
        .summary_table()) works exactly as after a sweep().

        Args:
            dim_filter: DimeFilter used — its .tag contributes to the filename
            importance: ImportanceMatrix produced by dim_filter.compute()
            k:          top-k documents to retrieve
            save:       if True, write the run to disk

        Returns:
            SweepResults with a single entry keyed by the selector tag
        """
        query_ids   = importance.query_ids
        qembs       = self.qrys_encoder.get_encoding(query_ids)
        alpha_key   = self.selector.tag          # e.g. "rdime" — survives CSV round-trip

        weights  = self.selector.select(importance, alpha=None)
        masked_q = (qembs * weights).astype(np.float32)
        scores, offsets = self.index.search(masked_q, k)

        run = self._build_run(
            query_ids, scores, offsets, k, alpha=alpha_key,
            filter_tag=dim_filter.tag,
            selector_tag=self.selector.tag,
        )

        # per-query fraction of retained dimensions — only meaningful for
        # alpha-free selectors like RDIME where every query has its own budget
        retained_fracs = pd.Series(
            weights.mean(axis=1),    # [N] — fraction of dims kept per query
            index=query_ids,
            name="retained_frac",
        )

        results = SweepResults(
            model_name=self.model_name,
            collection=self.collection,
            filter_tag=dim_filter.tag,
            selector_tag=self.selector.tag,
            data={alpha_key: run},
            retained_fracs={alpha_key: retained_fracs},
        )

        if save:
            results.save()

        return results

    # ── alpha sweep — search only, no evaluation ───────────────────────────────

    def sweep(
        self,
        dim_filter: DimeFilter,
        importance: ImportanceMatrix,
        alphas: Sequence[float] = DEFAULT_ALPHAS,
        k: int = 1000,
        n_jobs: int = 1,
        save: bool = False,
    ) -> "SweepResults":
        """
        Run weighted search for every alpha, optionally in parallel.
        Evaluation is NOT performed here — call results.evaluate() separately.

        Query embeddings are fetched once and shared across all threads.
        FAISS releases the GIL during index.search so ThreadPoolExecutor
        gives near-linear speedup up to len(alphas) workers.

        When save=True, all alpha runs are concatenated into one TSV:
            data/runs/{collection}/{model}__{filter.tag}__{selector.tag}.tsv
        with an `alpha` column to distinguish runs.

        Args:
            dim_filter: the DimeFilter used — its .tag contributes to the filename
            importance: ImportanceMatrix produced by dim_filter.compute()
            alphas:     sequence of alpha values in (0, 1]
            k:          top-k documents to retrieve
            n_jobs:     number of parallel threads; -1 = one thread per alpha
            save:       if True, write the combined run to disk after sweep

        Returns:
            SweepResults indexed by alpha — runs only, no metrics yet
        """
        query_ids = importance.query_ids
        qembs     = self.qrys_encoder.get_encoding(query_ids)   # [N, D] — fetched once
        n_workers = len(alphas) if n_jobs == -1 else n_jobs

        def _search_alpha(alpha: float):
            weights  = self.selector.select(importance, alpha)   # [N, D] float32
            masked_q = (qembs * weights).astype(np.float32)

            # FAISS releases the GIL here — threads run concurrently
            scores, offsets = self.index.search(masked_q, k)
            run = self._build_run(
                query_ids, scores, offsets, k, alpha,
                filter_tag=dim_filter.tag,
                selector_tag=self.selector.tag,
            )
            logger.info(f"alpha={alpha:.2f} | search done")
            return alpha, run

        # ── execute ────────────────────────────────────────────────────────────
        raw: list[tuple] = []

        if n_workers == 1:
            raw = [_search_alpha(a) for a in alphas]
        else:
            logger.info(f"Sweeping {len(alphas)} alphas with {n_workers} threads")
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                futures = {pool.submit(_search_alpha, a): a for a in alphas}
                for future in as_completed(futures):
                    raw.append(future.result())

        raw.sort(key=lambda t: t[0])

        results = SweepResults(
            model_name=self.model_name,
            collection=self.collection,
            filter_tag=dim_filter.tag,
            selector_tag=self.selector.tag,
            data={alpha: run for alpha, run in raw},
        )

        if save:
            results.save()

        return results

    # ── helpers ────────────────────────────────────────────────────────────────

    def _build_run(
        self,
        query_ids: list[str],
        scores: np.ndarray,
        offsets: np.ndarray,
        k: int,
        alpha: float,
        filter_tag: str = "",
        selector_tag: str = "",
    ) -> pd.DataFrame:
        n_queries = len(query_ids)
        parts     = [self.model_name, filter_tag, selector_tag]
        run_tag   = "__".join(p for p in parts if p)
        return pd.DataFrame({
            "query_id": np.repeat(query_ids, k),
            "Q0":       "Q0",
            "doc_id":   [self._offset_to_did[o] for o in offsets.ravel()],
            "rank":     np.tile(np.arange(k), n_queries),
            "score":    scores.ravel(),
            "run":      run_tag,
            "alpha":    alpha,
        })


# ── Results container ──────────────────────────────────────────────────────────

class SweepResults:
    """
    Container for the full alpha sweep output.

    Stores one run DataFrame per alpha. Evaluation is on demand —
    call .evaluate(qrels) to compute metrics without re-running search.

    File naming convention:
        data/runs/{collection}/{model}__{filter_tag}__{selector_tag}.tsv
        data/results/{collection}/{model}__{filter_tag}__{selector_tag}.csv

    This means runs from different selectors applied to the same filter
    are stored in separate files and never overwrite each other.

    Usage:
        # after sweep
        results = searcher.sweep(dim_filter, importance, save=True)

        # evaluate when needed
        results.evaluate(qrels)

        # access
        results.run(0.3)                # run DataFrame for alpha=0.3
        results.mean(0.3, "nDCG@10")   # scalar metric
        results.summary_table()         # alpha × measure DataFrame

        # load previously saved sweep (no need to re-run search)
        results = SweepResults.load("contriever", "dl19", "prf-k10", "top-alpha")
        results.evaluate(qrels)
    """

    def __init__(
        self,
        model_name: str,
        collection: str,
        filter_tag: str,
        selector_tag: str,
        data: dict[float, pd.DataFrame],                      # alpha → run DataFrame
        retained_fracs: dict | None = None,                   # alpha → Series[query_id → frac]
    ):
        self.model_name      = model_name
        self.collection      = collection
        self.filter_tag      = filter_tag
        self.selector_tag    = selector_tag
        self._data           = data                           # alpha → run DataFrame
        self._metrics        = {}                             # alpha → (per_query df, summary df)
        self._retained_fracs = retained_fracs or {}           # alpha → Series (optional)
        self.alphas          = sorted(data.keys(), key=lambda a: (isinstance(a, str), a))

    # ── filename stem — single source of truth ─────────────────────────────────

    @property
    def _stem(self) -> str:
        return f"{self.model_name}__{self.filter_tag}__{self.selector_tag}"

    # ── search results ─────────────────────────────────────────────────────────

    def run(self, alpha: float) -> pd.DataFrame:
        """Return the TREC run DataFrame for a given alpha."""
        if alpha not in self._data:
            raise KeyError(f"Alpha {alpha} not in sweep. Available: {self.alphas}")
        return self._data[alpha]

    # ── evaluation — on demand ─────────────────────────────────────────────────

    def evaluate(
        self,
        qrels: pd.DataFrame,
        measures: list[str] = DEFAULT_MEASURES,
    ) -> "SweepResults":
        """
        Evaluate all alpha runs against qrels and cache the results.
        Returns self so calls can be chained:
            results.evaluate(qrels).summary_table()

        Args:
            qrels:    relevance judgments DataFrame
            measures: list of measure strings e.g. ["nDCG@10", "AP"]
        """
        for alpha in self.alphas:
            per_query  = evaluate(self._data[alpha], qrels, measures)
            summary_df = summary(per_query)
            self._metrics[alpha] = (per_query, summary_df)
            ndcg     = summary_df.loc[summary_df["measure"] == "nDCG@10", "mean"]
            ndcg_val = ndcg.values[0] if len(ndcg) else float("nan")
            alpha_str = f"{alpha:.2f}" if isinstance(alpha, float) else str(alpha)
            logger.info(f"alpha={alpha_str} | nDCG@10={ndcg_val:.4f}")
        return self

    @property
    def has_metrics(self) -> bool:
        return len(self._metrics) > 0

    def _require_metrics(self):
        if not self.has_metrics:
            raise RuntimeError("No metrics available. Call .evaluate(qrels) first.")

    def per_query(self, alpha: float) -> pd.DataFrame:
        """Return per-query evaluation results for a given alpha."""
        self._require_metrics()
        return self._metrics[alpha][0]

    def mean(self, alpha: float, measure: str) -> float:
        """Return mean metric value for a given alpha and measure."""
        self._require_metrics()
        summ = self._metrics[alpha][1]
        row  = summ.loc[summ["measure"] == measure, "mean"]
        if row.empty:
            raise KeyError(f"Measure '{measure}' not found.")
        return float(row.values[0])

    def summary_table(self) -> pd.DataFrame:
        """
        Wide DataFrame — one row per alpha, one column per measure.
        Requires .evaluate(qrels) to have been called first.

            alpha | nDCG@10 | AP | R@1000 | RR@10
            0.1   | 0.631   | …
            0.2   | 0.648   | …
        """
        self._require_metrics()
        rows = []
        for alpha in self.alphas:
            summ = self._metrics[alpha][1]
            row  = {"alpha": alpha}
            row.update(dict(zip(summ["measure"], summ["mean"])))
            rows.append(row)
        return pd.DataFrame(rows).set_index("alpha")

    # ── persistence ────────────────────────────────────────────────────────────

    @property
    def path(self) -> Path:
        """Path for the saved sweep TSV."""
        return RUNS_DIR / self.collection / f"{self._stem}.tsv"

    @property
    def results_path(self) -> Path:
        """Path for the saved evaluation results CSV."""
        return DATA_DIR / "results" / self.collection / f"{self._stem}.csv"

    def save(self) -> Path:
        """
        Concatenate all alpha runs into one TSV and save to disk.

        Path:    data/runs/{collection}/{model}__{filter_tag}__{selector_tag}.tsv
        Columns: query_id, Q0, doc_id, rank, score, run, alpha
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        combined = pd.concat(list(self._data.values()), ignore_index=True)
        combined.to_csv(self.path, sep="\t", index=False)
        logger.info(f"Saved {len(self.alphas)} alpha runs ({len(combined)} rows) → {self.path}")
        return self.path

    def save_results(self) -> Path:
        """
        Save per-query evaluation results to disk.
        Requires .evaluate(qrels) to have been called first.

        Path:    data/results/{collection}/{model}__{filter_tag}__{selector_tag}.csv
        Columns: alpha, query_id, measure, value
        """
        self._require_metrics()
        self.results_path.parent.mkdir(parents=True, exist_ok=True)

        per_query_frames = []
        for alpha in self.alphas:
            pq = self._metrics[alpha][0].copy()
            pq["alpha"] = alpha
            per_query_frames.append(pq)

        combined = pd.concat(per_query_frames, ignore_index=True)

        # merge per-query retained_frac if available (e.g. from RDIME)
        if self._retained_fracs:
            frac_frames = []
            for alpha in self.alphas:
                if alpha in self._retained_fracs:
                    s = self._retained_fracs[alpha].rename("retained_frac").reset_index()
                    s.columns = ["query_id", "retained_frac"]
                    s["alpha"] = alpha
                    frac_frames.append(s)
            if frac_frames:
                fracs_df = pd.concat(frac_frames, ignore_index=True)
                combined = combined.merge(fracs_df, on=["query_id", "alpha"], how="left")

        combined.to_csv(self.results_path, index=False)
        logger.info(f"Saved evaluation results ({len(self.alphas)} alphas) → {self.results_path}")
        return self.results_path

    @classmethod
    def load(
        cls,
        model_name: str,
        collection: str,
        filter_tag: str,
        selector_tag: str,
    ) -> "SweepResults":
        """
        Load a previously saved sweep TSV back into a SweepResults object.
        Call .evaluate(qrels) afterwards to compute metrics.

        Usage:
            results = SweepResults.load("contriever", "dl19", "prf-k10", "top-alpha")
            results.evaluate(qrels)
            results.summary_table()
        """
        stem = f"{model_name}__{filter_tag}__{selector_tag}"
        path = RUNS_DIR / collection / f"{stem}.tsv"
        if not path.exists():
            raise FileNotFoundError(
                f"No saved sweep at {path}. Run sweep(save=True) first."
            )

        df     = pd.read_csv(path, sep="\t", dtype={"query_id": str, "doc_id": str})
        raw_alphas = df["alpha"].unique()
        # alpha column may be floats (top-alpha sweep) or a string (rdime single-shot)
        alphas = sorted(raw_alphas, key=lambda a: (isinstance(a, str), a))
        data   = {
            alpha: df[df["alpha"] == alpha].reset_index(drop=True)
            for alpha in alphas
        }
        logger.info(f"Loaded sweep: {path} | {len(alphas)} alphas")
        return cls(
            model_name=model_name,
            collection=collection,
            filter_tag=filter_tag,
            selector_tag=selector_tag,
            data=data,
        )

    def load_results(self) -> "SweepResults":
        """
        Load previously saved evaluation results back into _metrics.
        Allows inspecting metrics without re-running evaluate().

        Usage:
            results = SweepResults.load("contriever", "dl19", "prf-k10", "top-alpha")
            results.load_results()
            results.summary_table()
        """
        if not self.results_path.exists():
            raise FileNotFoundError(
                f"No saved results at {self.results_path}. "
                f"Call .evaluate(qrels) and .save_results() first."
            )

        df = pd.read_csv(self.results_path, dtype={"query_id": str})

        for alpha in sorted(df["alpha"].unique()):
            pq         = df[df["alpha"] == alpha][["query_id", "measure", "value"]].reset_index(drop=True)
            summary_df = summary(pq)
            self._metrics[alpha] = (pq, summary_df)

        logger.info(f"Loaded results: {self.results_path} | {len(self._metrics)} alphas")
        return self

    def __repr__(self) -> str:
        return (
            f"SweepResults("
            f"model={self.model_name}, "
            f"filter={self.filter_tag}, "
            f"selector={self.selector_tag}, "
            f"alphas={self.alphas}, "
            f"has_metrics={self.has_metrics})"
        )