"""FastAPI application: JSON API + the single-page web UI."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app import __version__
from app.config import get_settings
from app.db.models import Database
from app.pipeline import RAGPipeline
from app.schemas import AskRequest, AskResponse, ChunkOut, DocumentOut, FeedbackRequest

STATIC_DIR = Path(__file__).parent / "static"
logger = logging.getLogger("app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logging.basicConfig(level=settings.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    app.state.pipeline = RAGPipeline(settings)
    try:
        app.state.db = Database(settings.database_url)
    except Exception as exc:  # the assistant still works without persistence
        logger.error("Database unavailable, query logging disabled: %s", exc)
        app.state.db = None
    yield


app = FastAPI(
    title="Enterprise RAG Knowledge Assistant",
    description="Answers employee questions over company policy documents with cited sources.",
    version=__version__,
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def _pipeline(request: Request) -> RAGPipeline:
    return request.app.state.pipeline


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health(request: Request) -> dict:
    return {"status": "ok", "version": __version__, **_pipeline(request).describe(),
            "database": request.app.state.db is not None}


@app.post("/api/ask", response_model=AskResponse)
async def ask(body: AskRequest, request: Request) -> AskResponse:
    pipeline = _pipeline(request)
    result = await run_in_threadpool(pipeline.ask, body.question)

    query_id = None
    if (db := request.app.state.db) is not None:
        try:
            query_id = db.log_query(
                question=result.question,
                answer=result.answer,
                citations=[c["id"] for c in result.citations],
                confidence=result.confidence,
                used_fallback=result.used_fallback,
                latency_ms=result.latency_ms,
                mode=pipeline.mode,
            )
        except Exception as exc:
            logger.warning("Failed to log query: %s", exc)

    return AskResponse(
        query_id=query_id,
        question=result.question,
        answer=result.answer,
        citations=result.citations,
        related=result.related,
        confidence=result.confidence,
        grounded=result.grounded,
        used_fallback=result.used_fallback,
        queries=result.queries,
        mode=pipeline.mode,
        latency_ms=result.latency_ms,
        trace=result.trace if body.include_trace else [],
        warnings=result.warnings,
    )


@app.post("/api/feedback", status_code=201)
def feedback(body: FeedbackRequest, request: Request) -> dict:
    db = request.app.state.db
    if db is None:
        raise HTTPException(503, "Feedback storage is not available")
    if not db.add_feedback(body.query_id, body.rating, body.comment):
        raise HTTPException(404, "Unknown query_id")
    return {"status": "recorded"}


@app.get("/api/documents", response_model=list[DocumentOut])
def documents(request: Request) -> list[DocumentOut]:
    pipeline = _pipeline(request)
    counts: dict[str, int] = {}
    for c in pipeline.chunks:
        counts[c.doc_id] = counts.get(c.doc_id, 0) + 1
    return [
        DocumentOut(
            doc_id=d.doc_id,
            title=d.title,
            owner=d.metadata.get("owner"),
            version=d.metadata.get("version"),
            effective_date=d.metadata.get("effective_date"),
            source=d.source,
            sections=counts.get(d.doc_id, 0),
        )
        for d in pipeline.documents
    ]


@app.get("/api/chunks/{chunk_id}", response_model=ChunkOut)
def chunk(chunk_id: str, request: Request) -> ChunkOut:
    c = _pipeline(request).get_chunk(chunk_id)
    if c is None:
        raise HTTPException(404, f"Unknown section {chunk_id}")
    return ChunkOut(id=c.chunk_id, doc_id=c.doc_id, doc_title=c.doc_title, section=c.section,
                    heading=c.heading, breadcrumb=c.breadcrumb, text=c.text)
