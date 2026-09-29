"""Request/response models for the HTTP API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000, examples=["How many days of annual leave do I get?"])
    include_trace: bool = True


class CitationOut(BaseModel):
    id: str
    doc_id: str
    doc_title: str
    section: str
    heading: str
    snippet: str
    score: float | None = None


class TraceStepOut(BaseModel):
    node: str
    ms: float
    detail: str


class AskResponse(BaseModel):
    query_id: str | None
    question: str
    answer: str
    citations: list[CitationOut]
    related: list[CitationOut]
    confidence: Literal["high", "medium", "low"]
    grounded: bool
    used_fallback: bool
    queries: list[str]
    mode: Literal["mock", "live"]
    latency_ms: float
    trace: list[TraceStepOut] = []
    warnings: list[str] = []


class FeedbackRequest(BaseModel):
    query_id: str
    rating: Literal[1, -1]
    comment: str | None = Field(default=None, max_length=2000)


class DocumentOut(BaseModel):
    doc_id: str
    title: str
    owner: str | None
    version: str | None
    effective_date: str | None
    source: str
    sections: int


class ChunkOut(BaseModel):
    id: str
    doc_id: str
    doc_title: str
    section: str
    heading: str
    breadcrumb: str
    text: str
