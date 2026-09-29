"""LangGraph node implementations.

Flow:  analyze_query -> hybrid_retrieve -> rerank -> (route)
          route = generate          if best rerank score >= threshold
                = fallback_retrieve if not yet retried  (multi-query, wider, BM25-weighted)
                = abstain           otherwise
       fallback_retrieve -> rerank (loop, max 1 retry)
       generate -> validate_citations -> END
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable
from functools import wraps

from app.graph.state import Citation, GraphState
from app.llm.base import LLM
from app.llm.mock_llm import MockLLM
from app.llm.prompts import NOT_FOUND_ANSWER
from app.retrieval.hybrid import HybridRetriever, RetrievedChunk
from app.retrieval.reranker import Reranker, rerank

logger = logging.getLogger(__name__)

CITATION_PATTERN = re.compile(r"\[([A-Z]{2,5}-\d+(?:\.\d+)*(?:\.p\d+)?)\]")
MAX_FALLBACK_ATTEMPTS = 1
ABSTAIN_MESSAGE = (
    "I couldn't find a policy that answers this question. "
    "Please contact People Operations (peopleops@northwind.example) or the relevant policy owner."
)


def traced(name: str) -> Callable:
    """Record per-node latency plus a short human-readable detail in `state["trace"]`."""

    def decorator(fn: Callable[..., dict]) -> Callable[..., dict]:
        @wraps(fn)
        def wrapper(self, state: GraphState) -> dict:
            start = time.perf_counter()
            update = fn(self, state)
            detail = update.pop("_detail", "")
            ms = round((time.perf_counter() - start) * 1000, 1)
            update["trace"] = [{"node": name, "ms": ms, "detail": detail}]
            logger.debug("%s (%.1f ms) %s", name, ms, detail)
            return update

        return wrapper

    return decorator


def to_citation(rc: RetrievedChunk, snippet_chars: int = 320) -> Citation:
    c = rc.chunk
    snippet = c.text if len(c.text) <= snippet_chars else c.text[:snippet_chars].rsplit(" ", 1)[0] + "…"
    return Citation(
        id=c.chunk_id,
        doc_id=c.doc_id,
        doc_title=c.doc_title,
        section=c.section,
        heading=c.heading,
        snippet=snippet,
        score=rc.rerank_score,
    )


def validate_citations(answer: str, contexts: list[RetrievedChunk]) -> tuple[str, list[Citation], list[str]]:
    """Keep citations that point at retrieved context; strip any the model invented."""
    by_id = {rc.chunk.chunk_id: rc for rc in contexts}
    cited, invalid = [], []
    for cid in CITATION_PATTERN.findall(answer):
        if cid in by_id:
            if cid not in cited:
                cited.append(cid)
        elif cid not in invalid:
            invalid.append(cid)
    for cid in invalid:
        answer = answer.replace(f"[{cid}]", "")
    answer = re.sub(r"\s+([.,;:])", r"\1", re.sub(r"[ \t]{2,}", " ", answer)).strip()
    return answer, [to_citation(by_id[cid]) for cid in cited], invalid


class GraphNodes:
    def __init__(
        self,
        retriever: HybridRetriever,
        reranker: Reranker,
        llm: LLM,
        top_k_retrieve: int = 20,
        top_k_rerank: int = 5,
        threshold: float | None = None,
    ):
        self.retriever = retriever
        self.reranker = reranker
        self.llm = llm
        self.top_k_retrieve = top_k_retrieve
        self.top_k_rerank = top_k_rerank
        self.threshold = threshold if threshold is not None else reranker.default_threshold

    # ------------------------------------------------------------------ nodes
    @traced("analyze_query")
    def analyze_query(self, state: GraphState) -> dict:
        question = re.sub(r"\s+", " ", state["question"]).strip()
        return {
            "question": question,
            "queries": [question],
            "attempts": 0,
            "used_fallback": False,
            "threshold": self.threshold,
            "_detail": f"{len(question.split())} words",
        }

    @traced("hybrid_retrieve")
    def hybrid_retrieve(self, state: GraphState) -> dict:
        candidates = self.retriever.search(state["queries"], k=self.top_k_retrieve)
        both = sum(1 for c in candidates if c.bm25_rank and c.vector_rank)
        return {
            "candidates": candidates,
            "_detail": f"{len(candidates)} candidates ({both} found by both BM25 and vector)",
        }

    @traced("rerank")
    def rerank(self, state: GraphState) -> dict:
        reranked = rerank(self.reranker, state["queries"], state["candidates"], self.top_k_rerank)
        top = reranked[0].rerank_score if reranked else 0.0
        return {
            "reranked": reranked,
            "top_score": top,
            "_detail": f"{self.reranker.name}: top={top:.3f} threshold={self.threshold:.2f}",
        }

    @traced("fallback_retrieve")
    def fallback_retrieve(self, state: GraphState) -> dict:
        question = state["question"]
        warnings: list[str] = []
        try:
            rewrites = self.llm.rewrite_queries(question)
        except Exception as exc:
            logger.warning("Query rewrite failed: %s", exc)
            rewrites, warnings = MockLLM().rewrite_queries(question), ["query_rewrite_degraded"]
        queries = list(dict.fromkeys([question, *rewrites]))
        # Wider net, and lean on exact keyword matches, which recover well from vocabulary gaps.
        candidates = self.retriever.search(
            queries, k=self.top_k_retrieve * 2, bm25_weight=1.5, vector_weight=1.0
        )
        return {
            "queries": queries,
            "candidates": candidates,
            "attempts": state.get("attempts", 0) + 1,
            "used_fallback": True,
            "warnings": warnings,
            "_detail": f"{len(queries)} queries: " + " | ".join(queries[1:]),
        }

    @traced("generate")
    def generate(self, state: GraphState) -> dict:
        contexts = self._answer_contexts(state)
        warnings: list[str] = []
        try:
            answer = self.llm.generate_answer(state["question"], contexts)
        except Exception as exc:  # keep serving answers if the LLM provider is down
            logger.error("LLM generation failed, using extractive fallback: %s", exc)
            answer = MockLLM().generate_answer(state["question"], contexts)
            warnings.append("llm_unavailable_extractive_answer")
        return {"answer": answer, "warnings": warnings, "_detail": f"{len(contexts)} context sections"}

    @traced("validate_citations")
    def validate_citations(self, state: GraphState) -> dict:
        contexts = self._answer_contexts(state)
        answer, citations, invalid = validate_citations(state["answer"], contexts)
        warnings = [f"removed_invalid_citation:{cid}" for cid in invalid]

        if answer.startswith(NOT_FOUND_ANSWER.rstrip(".")):
            return {
                "answer": ABSTAIN_MESSAGE,
                "citations": [],
                "related": self._related(state),
                "confidence": "low",
                "grounded": True,
                "warnings": warnings,
                "_detail": "model reported answer not in context",
            }

        grounded = bool(citations)
        if not grounded:
            warnings.append("answer_has_no_valid_citations")
        confidence = "high" if grounded and not state.get("used_fallback") else "medium"
        if not grounded:
            confidence = "low"
        return {
            "answer": answer,
            "citations": citations,
            "related": [],
            "confidence": confidence,
            "grounded": grounded,
            "warnings": warnings,
            "_detail": f"{len(citations)} valid, {len(invalid)} removed",
        }

    @traced("abstain")
    def abstain(self, state: GraphState) -> dict:
        return {
            "answer": ABSTAIN_MESSAGE,
            "citations": [],
            "related": self._related(state),
            "confidence": "low",
            "grounded": True,
            "_detail": f"best score {state.get('top_score', 0):.3f} below threshold",
        }

    # ---------------------------------------------------------------- routing
    def route_after_rerank(self, state: GraphState) -> str:
        if state.get("top_score", 0.0) >= self.threshold:
            return "generate"
        if state.get("attempts", 0) < MAX_FALLBACK_ATTEMPTS:
            return "fallback"
        return "abstain"

    # ---------------------------------------------------------------- helpers
    def _answer_contexts(self, state: GraphState) -> list[RetrievedChunk]:
        """Sections passed to the LLM: those reasonably close to the best match (always >= 1)."""
        reranked = state.get("reranked", [])
        floor = self.threshold * 0.5
        contexts = [rc for rc in reranked if (rc.rerank_score or 0) >= floor]
        return contexts or reranked[:1]

    def _related(self, state: GraphState, limit: int = 3) -> list[Citation]:
        """Nearest sections to offer when abstaining; skip ones with no real overlap."""
        floor = self.threshold * 0.25
        return [to_citation(rc) for rc in state.get("reranked", [])[:limit] if (rc.rerank_score or 0) >= floor]
