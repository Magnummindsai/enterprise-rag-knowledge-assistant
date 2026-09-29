"""Shared state passed between LangGraph nodes."""

from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

from app.retrieval.hybrid import RetrievedChunk

Confidence = Literal["high", "medium", "low"]


class TraceStep(TypedDict):
    node: str
    ms: float
    detail: str


class Citation(TypedDict):
    id: str
    doc_id: str
    doc_title: str
    section: str
    heading: str
    snippet: str
    score: float | None


class GraphState(TypedDict, total=False):
    question: str
    queries: list[str]                # queries used for the latest retrieval pass
    candidates: list[RetrievedChunk]  # fused hybrid results
    reranked: list[RetrievedChunk]    # top-k after cross-encoder
    top_score: float
    threshold: float
    attempts: int                     # number of fallback passes performed
    used_fallback: bool
    answer: str
    citations: list[Citation]
    related: list[Citation]           # nearest sources when the assistant abstains
    confidence: Confidence
    grounded: bool
    warnings: Annotated[list[str], operator.add]
    trace: Annotated[list[TraceStep], operator.add]
