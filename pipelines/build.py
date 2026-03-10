# pipelines/build.py
#
# Single entry point for encoding corpus, queries and building the FAISS index.
# Steps are skipped automatically if outputs already exist.
#
# Usage:
#   python pipelines/build.py --collection dl19 --model contriever
#   python pipelines/build.py --collection dl19 --model contriever --overwrite
#   python pipelines/build.py --collection dl19 --model contriever --skip_corpus --skip_index

import argparse
import logging
from dotenv import load_dotenv

load_dotenv()

import sys
sys.path.append('/hdd4/giuder/Documents/projects/dime')

from src.config import COLLECTIONS, MODEL_TO_HF
from src.data_loading import CollectionLoader
from src.encode import encode_corpus, encode_queries
from src.index import build_index

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Encode corpus + queries and build FAISS index for a given collection and model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-c", "--collection",
        required=True,
        choices=list(COLLECTIONS.keys()),
        help="Test collection to build. e.g. dl19, dl20, robust04, antique",
    )
    parser.add_argument(
        "-m", "--model",
        required=True,
        choices=list(MODEL_TO_HF.keys()),
        help="Retrieval model to use for encoding. e.g. contriever, tasb, ance",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-encode and rebuild everything even if outputs already exist.",
    )
    parser.add_argument(
        "--skip_corpus",
        action="store_true",
        help="Skip corpus encoding (e.g. already encoded from a previous collection).",
    )
    parser.add_argument(
        "--skip_queries",
        action="store_true",
        help="Skip query encoding.",
    )
    parser.add_argument(
        "--skip_index",
        action="store_true",
        help="Skip FAISS index building.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    logger.info(f"Collection : {args.collection}")
    logger.info(f"Model      : {args.model}")
    logger.info(f"Overwrite  : {args.overwrite}")

    loader = CollectionLoader(args.collection)
    logger.info(f"Loader     : {loader}")

    # ── Corpus ────────────────────────────────────────────────────────────────
    if not args.skip_corpus:
        logger.info("── Step 1/3: Encoding corpus ──")
        encode_corpus(
            docs=loader.corpus(),
            model_name=args.model,
            collection=args.collection,
            overwrite=args.overwrite,
        )
    else:
        logger.info("── Step 1/3: Corpus encoding skipped ──")

    # ── Queries ───────────────────────────────────────────────────────────────
    if not args.skip_queries:
        logger.info("── Step 2/3: Encoding queries ──")
        encode_queries(
            queries=loader.queries(),
            model_name=args.model,
            collection=args.collection,
            overwrite=args.overwrite,
        )
    else:
        logger.info("── Step 2/3: Query encoding skipped ──")

    # ── FAISS index ───────────────────────────────────────────────────────────
    if not args.skip_index:
        logger.info("── Step 3/3: Building FAISS index ──")
        build_index(
            model_name=args.model,
            collection=args.collection,
            overwrite=args.overwrite,
        )
    else:
        logger.info("── Step 3/3: Index building skipped ──")

    logger.info("── Build complete ──")


if __name__ == "__main__":
    main()