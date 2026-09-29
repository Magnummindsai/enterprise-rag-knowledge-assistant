"""Sparse keyword retrieval with BM25 (rank_bm25)."""

from __future__ import annotations

import numpy as np
from rank_bm25 import BM25Okapi

from app.ingestion.chunker import Chunk
from app.retrieval.text import tokenize


class BM25Index:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self._bm25 = BM25Okapi([tokenize(c.index_text) for c in chunks])

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        """Return (chunk_index, score) pairs for the top-k chunks with a positive score."""
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        order = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in order if scores[i] > 0]
