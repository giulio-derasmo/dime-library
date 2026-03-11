# pipelines/dime.py
#
# DIME masked retrieval pipeline.
# Computes dimension importance, runs weighted search over alpha grid
# (or a single shot for alpha-free selectors), evaluates and saves results.
#
# Usage:
#   # Standard alpha sweep (top-alpha selector, default)
#   python pipelines/dime.py -c dl19 -m contriever -f prf --config configs/prf_k10.yaml
#
#   # RDIME — single run, no alpha grid, no hyperparameters
#   python pipelines/dime.py -c dl19 -m contriever -f prf --config configs/prf_k10.yaml --selector rdime
#
#   # Parallel sweep
#   python pipelines/dime.py -c dl19 -m contriever -f prf --config configs/prf_k10.yaml --n_jobs -1
#
#   # Oracular upper bound with RDIME threshold
#   python pipelines/dime.py -c dl19 -m contriever -f oracular --config configs/oracular.yaml --selector rdime

import argparse
import logging

import yaml

from dotenv import load_dotenv
load_dotenv()

import sys
sys.path.append('/hdd4/giuder/Documents/projects/dime')

from src.config import COLLECTIONS, MODEL_TO_HF, DEFAULT_MEASURES, RUNS_DIR, FILTERS, SELECTORS
from src.data_loading import CollectionLoader
from src.evaluate import load_run
from src.index import load_index
from src.memmap_interface import CorpusEncoding, CorpusMapping, QueriesEncoding
from src.dime.masked_search import MaskedSearcher, SweepResults, DEFAULT_ALPHAS
from src.dime.selectors import TopAlphaSelector, RDIMESelector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="DIME masked retrieval pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-c", "--collection", required=True,  choices=list(COLLECTIONS))
    parser.add_argument("-m", "--model",      required=True,  choices=list(MODEL_TO_HF))
    parser.add_argument("-f", "--filter",     required=True,  choices=FILTERS)
    parser.add_argument("-s", "--selector",   default="top-alpha", choices=SELECTORS,
                        help=(
                            "Dimension selection strategy. "
                            "'top-alpha' sweeps a fixed fraction of dims (requires --alphas). "
                            "'rdime' derives a per-query threshold from the data — no alpha needed."
                        ))
    parser.add_argument("--config",   required=True,
                        help="Path to filter config YAML e.g. configs/prf_k10.yaml")
    parser.add_argument("--alphas",   default=None, nargs="+", type=float,
                        help="Alpha values to sweep. Ignored when --selector rdime.")
    parser.add_argument("--k",        default=1000, type=int,
                        help="Docs retrieved per query")
    parser.add_argument("--n_jobs",   default=1,    type=int,
                        help="Parallel threads for sweep (-1 = one per alpha). Ignored for rdime.")
    parser.add_argument("--measures", default=None, nargs="+",
                        help="Evaluation measures")
    parser.add_argument("--save",      action="store_true",
                        help="Save run(s) to disk")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-run even if output already exists on disk")
    return parser.parse_args()


def build_filter(args, filter_cfg, qrys_encoder, qrels):
    """Instantiate the requested DimeFilter from CLI args and config."""
    if args.filter == "prf":
        from src.dime.filters.prf import PRFFilter
        docs_encoder = CorpusEncoding(args.model, args.collection)
        run          = load_run(args.model, args.collection)
        return PRFFilter(
            qrys_encoder=qrys_encoder,
            docs_encoder=docs_encoder,
            run=run,
            k=filter_cfg["k"],
        )
    if args.filter == "oracular":
        pass
    if args.filter == "gpt":
        from src.dime.filters import GPTFilter
        return GPTFilter(
            qrys_encoder=qrys_encoder,
            model_name=args.model,
            collection=args.collection,
            variant=filter_cfg["variant"],   # from your YAML config
        )
    raise ValueError(f"Unknown filter: {args.filter}")


def build_selector(selector_name: str, qrys_encoder: QueriesEncoding, query_ids: list[str]):
    """
    Instantiate the requested DimSelector.

    RDIMESelector needs the raw query embeddings at construction time so it
    can estimate the per-query noise threshold eps_hat^2. We pass them here
    from the already-loaded QueriesEncoding to avoid reloading.
    """
    if selector_name == "top-alpha":
        return TopAlphaSelector()

    if selector_name == "rdime":
        qembs = qrys_encoder.get_encoding(query_ids)   # [N, D] — fetched once
        return RDIMESelector(qembs)

    # ── add new selectors here ─────────────────────────────────────────────────
    raise ValueError(f"Unknown selector: {selector_name!r}. Available: {SELECTORS}")


def main():
    args     = parse_args()
    measures = args.measures or DEFAULT_MEASURES

    with open(args.config) as f:
        filter_cfg = yaml.safe_load(f) or {}

    logger.info(f"Collection : {args.collection}")
    logger.info(f"Model      : {args.model}")
    logger.info(f"Filter     : {args.filter}")
    logger.info(f"Selector   : {args.selector}")
    logger.info(f"Config     : {filter_cfg}")
    logger.info(f"Measures   : {measures}")

    # ── load data ──────────────────────────────────────────────────────────────
    loader    = CollectionLoader(args.collection)
    queries   = loader.queries()
    qrels     = loader.qrels()
    query_ids = queries["query_id"].tolist()

    # ── load encodings ─────────────────────────────────────────────────────────
    qrys_encoder   = QueriesEncoding(args.model, args.collection)
    corpus_mapping = CorpusMapping(args.model, args.collection)
    index          = load_index(args.model, args.collection)

    # ── build filter + selector ────────────────────────────────────────────────
    dim_filter = build_filter(args, filter_cfg, qrys_encoder, qrels)
    selector   = build_selector(args.selector, qrys_encoder, query_ids)
    logger.info(f"Filter     : {dim_filter}")
    logger.info(f"Selector   : {selector}")

    # ── compute importance ─────────────────────────────────────────────────────
    logger.info("── Computing dimension importance ──")
    importance = dim_filter.compute(queries)
    logger.info(f"Importance : {importance}")

    # ── build searcher ─────────────────────────────────────────────────────────
    searcher = MaskedSearcher(
        index=index,
        qrys_encoder=qrys_encoder,
        corpus_mapping=corpus_mapping,
        model_name=args.model,
        collection=args.collection,
        selector=selector,
    )

    run_path = RUNS_DIR / args.collection / f"{args.model}__{dim_filter.tag}__{selector.tag}.tsv"

    # ── RDIME branch: single shot, no alpha grid ───────────────────────────────
    if args.selector == "rdime":
        if run_path.exists() and not args.overwrite:
            logger.info(f"Run already exists at {run_path} — loading from disk. Use --overwrite to re-run.")
            results = SweepResults.load(args.model, args.collection, dim_filter.tag, selector.tag)
        else:
            mean_frac = selector.mean_retained_frac(importance)
            logger.info(f"── Running RDIME single-shot search (mean retained dims: {mean_frac:.1%}) ──")
            results = searcher.run_once(
                dim_filter=dim_filter,
                importance=importance,
                k=args.k,
                save=True,
            )

    # ── Top-alpha branch: sweep over alpha grid ────────────────────────────────
    else:
        alphas = args.alphas or DEFAULT_ALPHAS
        logger.info(f"Alphas     : {alphas}")
        if run_path.exists() and not args.overwrite:
            logger.info(f"Sweep already exists at {run_path} — loading from disk. Use --overwrite to re-run.")
            results = SweepResults.load(args.model, args.collection, dim_filter.tag, selector.tag)
        else:
            logger.info("── Running masked search sweep ──")
            results = searcher.sweep(
                dim_filter=dim_filter,
                importance=importance,
                alphas=alphas,
                k=args.k,
                n_jobs=args.n_jobs,
                save=True,
            )

    # ── evaluate ───────────────────────────────────────────────────────────────
    logger.info("── Evaluating ──")
    results.evaluate(qrels, measures=measures)
    results.save_results()

    # ── print summary ──────────────────────────────────────────────────────────
    print(f"\n── {args.model} | {args.filter} | {args.selector} | {args.collection} ──")
    print(results.summary_table().to_string(float_format="{:.4f}".format))
    print()


if __name__ == "__main__":
    main()