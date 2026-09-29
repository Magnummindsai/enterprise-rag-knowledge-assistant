"""Wires ingestion, retrieval, reranking, the LLM and the LangGraph workflow together."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from app.config import Settings
from app.graph.nodes import GraphNodes
from app.graph.state import Citation, TraceStep
from app.graph.workflow import build_graph
from app.ingestion.chunker import Chunk, chunk_documents
from app.ingestion.loader import PolicyDocument, load_documents
from app.llm.base import get_llm
from app.retrieval.bm25 import BM25Index
from app.retrieval.embeddings import get_embedder
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.reranker import get_reranker
from app.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class AskResult:
    question: str
    answer: str
    citations: list[Citation]
    related: list[Citation]
    confidence: str
    grounded: bool
    used_fallback: bool
    queries: list[str]
    contexts: list[str]          # full text of reranked sections (used by evaluation)
    context_ids: list[str]
    latency_ms: float
    trace: list[TraceStep] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class RAGPipeline:
    def __init__(self, settings: Settings):
        start = time.perf_counter()
        self.settings = settings
        self.documents: list[PolicyDocument] = load_documents(settings.data_dir)
        self.chunks: list[Chunk] = chunk_documents(self.documents)

        embedder = get_embedder(settings)
        vectors = VectorStore.load_or_build(self.chunks, embedder, settings.index_dir)
        self.retriever = HybridRetriever(self.chunks, BM25Index(self.chunks), vectors)
        self.reranker = get_reranker(settings, self.chunks)
        self.llm = get_llm(settings)
        self.nodes = GraphNodes(
            self.retriever,
            self.reranker,
            self.llm,
            top_k_retrieve=settings.top_k_retrieve,
            top_k_rerank=settings.top_k_rerank,
            threshold=settings.relevance_threshold,
        )
        self.graph = build_graph(self.nodes)
        self.embedder_name = embedder.name
        self._chunk_by_id = {c.chunk_id: c for c in self.chunks}
        logger.info(
            "Pipeline ready in %.0f ms | mode=%s docs=%d chunks=%d embedder=%s reranker=%s llm=%s",
            (time.perf_counter() - start) * 1000, self.mode, len(self.documents), len(self.chunks),
            embedder.name, self.reranker.name, self.llm.name,
        )

    @property
    def mode(self) -> str:
        return "mock" if self.settings.is_mock else "live"

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        return self._chunk_by_id.get(chunk_id)

    def ask(self, question: str) -> AskResult:
        start = time.perf_counter()
        state = self.graph.invoke({"question": question, "trace": [], "warnings": []})
        reranked = state.get("reranked", [])
        return AskResult(
            question=state["question"],
            answer=state["answer"],
            citations=state.get("citations", []),
            related=state.get("related", []),
            confidence=state.get("confidence", "low"),
            grounded=state.get("grounded", False),
            used_fallback=state.get("used_fallback", False),
            queries=state.get("queries", [question]),
            contexts=[rc.chunk.text for rc in reranked],
            context_ids=[rc.chunk.chunk_id for rc in reranked],
            latency_ms=round((time.perf_counter() - start) * 1000, 1),
            trace=state.get("trace", []),
            warnings=state.get("warnings", []),
        )

    def describe(self) -> dict:
        return {
            "mode": self.mode,
            "llm": self.llm.name,
            "embedder": self.embedder_name,
            "reranker": self.reranker.name,
            "relevance_threshold": self.nodes.threshold,
            "documents": len(self.documents),
            "chunks": len(self.chunks),
        }
