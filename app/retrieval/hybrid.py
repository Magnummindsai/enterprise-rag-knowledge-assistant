"""Hybrid retrieval: BM25 + FAISS results merged with Reciprocal Rank Fusion (RRF)."""

from __future__ import annotations

from dataclasses import dataclass, replace

from app.ingestion.chunker import Chunk
from app.retrieval.bm25 import BM25Index
from app.retrieval.vector_store import VectorStore

RRF_K = 60  # standard constant from Cormack et al. (2009)


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    fused_score: float
    bm25_rank: int | None = None
    vector_rank: int | None = None
    rerank_score: float | None = None

    def with_rerank(self, score: float) -> RetrievedChunk:
        return replace(self, rerank_score=score)


def reciprocal_rank_fusion(
    ranked_lists: list[tuple[list[int], float]], k: int = RRF_K
) -> dict[int, float]:
    """Fuse ranked lists of ids. Each list comes with a weight; score = sum(w / (k + rank))."""
    scores: dict[int, float] = {}
    for ids, weight in ranked_lists:
        for rank, idx in enumerate(ids, start=1):
            scores[idx] = scores.get(idx, 0.0) + weight / (k + rank)
    return scores


class HybridRetriever:
    def __init__(self, chunks: list[Chunk], bm25: BM25Index, vectors: VectorStore):
        self.chunks = chunks
        self.bm25 = bm25
        self.vectors = vectors

    def search(
        self,
        queries: str | list[str],
        k: int = 20,
        bm25_weight: float = 1.0,
        vector_weight: float = 1.0,
        depth: int = 50,
    ) -> list[RetrievedChunk]:
        """Run every query through both retrievers and fuse all ranked lists with RRF."""
        if isinstance(queries, str):
            queries = [queries]
        ranked_lists: list[tuple[list[int], float]] = []
        best_bm25: dict[int, int] = {}
        best_vec: dict[int, int] = {}

        for query in queries:
            bm25_ids = [i for i, _ in self.bm25.search(query, depth)]
            vec_ids = [i for i, _ in self.vectors.search(query, depth)]
            ranked_lists += [(bm25_ids, bm25_weight), (vec_ids, vector_weight)]
            for rank, i in enumerate(bm25_ids, start=1):
                best_bm25[i] = min(best_bm25.get(i, rank), rank)
            for rank, i in enumerate(vec_ids, start=1):
                best_vec[i] = min(best_vec.get(i, rank), rank)

        fused = reciprocal_rank_fusion(ranked_lists)
        top = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:k]
        return [
            RetrievedChunk(
                chunk=self.chunks[i],
                fused_score=score,
                bm25_rank=best_bm25.get(i),
                vector_rank=best_vec.get(i),
            )
            for i, score in top
        ]
