# pipelines/search.py

import argparse
import logging
from dotenv import load_dotenv

load_dotenv()

import sys
sys.path.append('/hdd4/giuder/Documents/projects/dime')

from src.config import COLLECTIONS, MODEL_TO_HF, DEFAULT_MEASURES, RUNS_DIR
from src.data_loading import CollectionLoader
from src.evaluate import evaluate, summary, load_run, save_results
from src.search import search, save_run

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Baseline retrieval — search and optionally evaluate.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-c", "--collection", required=True, choices=list(COLLECTIONS))
    parser.add_argument("-m", "--model",      required=True, choices=list(MODEL_TO_HF))
    parser.add_argument("-k", "--k",          default=1000,  type=int,
                        help="Number of documents to retrieve per query")
    parser.add_argument("--evaluate",     action="store_true",
                        help="Evaluate the run after searching")
    parser.add_argument("--overwrite",    action="store_true",
                        help="Re-run search even if run already exists on disk")
    parser.add_argument("--measures",     nargs="+", default=DEFAULT_MEASURES,
                        help="Measures to compute e.g. --measures nDCG@10 AP")
    #parser.add_argument("--save_results", action="store_true",
    #                    help="Save per-query evaluation results to disk")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # ── search ─────────────────────────────────────────────────────────────────
    run_path = RUNS_DIR / args.collection / f"{args.model}.tsv"

    if run_path.exists() and not args.overwrite:
        logger.info(f"Run already exists at {run_path} — loading from disk. Use --overwrite to re-run.")
        run = load_run(args.model, args.collection)
    else:
        run = search(model_name=args.model, collection=args.collection, k=args.k)
        save_run(run, model_name=args.model, collection=args.collection)

    # ── evaluate ───────────────────────────────────────────────────────────────
    if args.evaluate:
        qrels   = CollectionLoader(args.collection).qrels()
        results = evaluate(run, qrels, measures=args.measures)

        print(f"\n── {args.model} | {args.collection} ──────────────────────────")
        print(summary(results).to_string(index=False, float_format="{:.4f}".format))
        print()

        save_results(results, args.model, args.collection)