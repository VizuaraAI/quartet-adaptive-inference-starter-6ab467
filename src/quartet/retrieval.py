"""BM25 retrieval over a Wikipedia paragraph corpus.

BM25 rather than a dense retriever on purpose: it costs almost no GPU memory,
so the RETRIEVE action's memory cost is dominated by the longer context rather
than by a second encoder. That keeps the four action costs cleanly separable,
which is what the memory objective in the paper is measuring. Swapping in a
dense retriever is a good week-7 extension, and the interface below is all you
need to reimplement.
"""

from __future__ import annotations

import os
import pickle
from typing import Sequence

WIKI_SPEC = ("rag-datasets/rag-mini-wikipedia", "text-corpus", "passages")


class BM25Retriever:
    """Thin wrapper around `bm25s` with an on-disk cache of the index."""

    def __init__(self, index_dir: str, max_passage_tokens: int = 160):
        self.index_dir = index_dir
        self.max_passage_tokens = max_passage_tokens
        self._retriever = None
        self._corpus: list[str] = []

    # ------------------------------------------------------------- build ---

    def build(self, passages: Sequence[str] | None = None) -> "BM25Retriever":
        import bm25s

        if passages is None:
            passages = self._load_default_corpus()
        self._corpus = [self._truncate(p) for p in passages if p and p.strip()]
        tokens = bm25s.tokenize(self._corpus, stopwords="en")
        retriever = bm25s.BM25()
        retriever.index(tokens)

        os.makedirs(self.index_dir, exist_ok=True)
        retriever.save(self.index_dir)
        with open(os.path.join(self.index_dir, "corpus.pkl"), "wb") as handle:
            pickle.dump(self._corpus, handle)
        self._retriever = retriever
        return self

    def _load_default_corpus(self) -> list[str]:
        from datasets import load_dataset

        repo, subset, split = WIKI_SPEC
        data = load_dataset(repo, subset, split=split)
        column = "passage" if "passage" in data.column_names else data.column_names[0]
        return [row[column] for row in data]

    def _truncate(self, passage: str) -> str:
        words = passage.split()
        if len(words) <= self.max_passage_tokens:
            return passage
        return " ".join(words[: self.max_passage_tokens])

    # -------------------------------------------------------------- load ---

    def load(self) -> "BM25Retriever":
        import bm25s

        corpus_path = os.path.join(self.index_dir, "corpus.pkl")
        if not os.path.exists(corpus_path):
            raise FileNotFoundError(
                f"no index at {self.index_dir}; run `scripts/build_index.sh` first"
            )
        self._retriever = bm25s.BM25.load(self.index_dir)
        with open(corpus_path, "rb") as handle:
            self._corpus = pickle.load(handle)
        return self

    def load_or_build(self) -> "BM25Retriever":
        try:
            return self.load()
        except (FileNotFoundError, OSError):
            return self.build()

    # ------------------------------------------------------------ search ---

    def search(self, query: str, top_k: int = 3) -> list[str]:
        import bm25s

        if self._retriever is None:
            self.load_or_build()
        k = min(top_k, len(self._corpus))
        if k == 0:
            return []
        indices, _scores = self._retriever.retrieve(
            bm25s.tokenize([query], stopwords="en"), k=k
        )
        return [self._corpus[int(i)] for i in indices[0]]


def build_retriever(cfg) -> BM25Retriever:
    return BM25Retriever(cfg.retrieval.index_dir, cfg.retrieval.max_passage_tokens)
