import ir_datasets
import pandas as pd
from tqdm import tqdm
tqdm.pandas()

from src.normalize_text import normalize
from src.config import (
    COLLECTIONS,
    get_corpus_name,
    get_ir_dataset_docs,
    get_ir_dataset_queries_qrels,
)


class CollectionLoader:
    """
    Single entry point for loading any data related to a test collection.
    Guarantees consistency: corpus, queries, and qrels all come from
    the same collection config.

    Usage:
        loader = CollectionLoader("dl19")
        corpus  = loader.corpus()
        queries = loader.queries()
        qrels   = loader.qrels()
    """

    def __init__(self, collection: str):
        if collection not in COLLECTIONS:
            raise ValueError(f"Unknown collection '{collection}'. Available: {list(COLLECTIONS)}")
        self.collection   = collection
        self.corpus_name  = get_corpus_name(collection)
        self._corpus_dataset  = ir_datasets.load(get_ir_dataset_docs(collection))
        self._qq_dataset      = ir_datasets.load(get_ir_dataset_queries_qrels(collection))

    def corpus(self) -> pd.DataFrame:
        doc_ids, texts = zip(*[
            (doc.doc_id, doc.text)
            for doc in tqdm(self._corpus_dataset.docs_iter(), desc=f"Loading corpus ({self.corpus_name})")
        ])
        docs = pd.DataFrame({"did": doc_ids, "text": texts})
        
        docs["text"] = docs["text"].progress_apply(normalize)
        return docs

    def queries(self) -> pd.DataFrame:
        """Test queries for this collection."""
        return pd.DataFrame([
            {"query_id": q.query_id, "text": q.text}
            for q in self._qq_dataset.queries_iter()
        ])

    def qrels(self) -> pd.DataFrame:
        """Relevance judgments for this collection."""
        return pd.DataFrame([
            {"query_id": qrel.query_id, "doc_id": qrel.doc_id, "relevance": qrel.relevance}
            for qrel in self._qq_dataset.qrels_iter()
        ])

    def all(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load corpus, queries and qrels in one call."""
        return self.corpus(), self.queries(), self.qrels()

    def __repr__(self):
        return f"CollectionLoader(collection={self.collection}, corpus={self.corpus_name})"