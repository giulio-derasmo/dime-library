# pipelines/search.py

import argparse
import logging
from dotenv import load_dotenv

load_dotenv()


import sys
sys.path.append('/hdd4/giuder/Documents/projects/dime')

from src.config import COLLECTIONS, MODEL_TO_HF
from src.search import search, save_run

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--collection", required=True, choices=list(COLLECTIONS))
    parser.add_argument("-m", "--model",      required=True, choices=list(MODEL_TO_HF))
    parser.add_argument("-k", "--k",          default=1000, type=int)
    args = parser.parse_args()

    run = search(model_name=args.model, collection=args.collection, k=args.k)
    save_run(run, model_name=args.model, collection=args.collection)