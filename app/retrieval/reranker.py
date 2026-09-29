"""Second-stage reranking.

* ``CrossEncoderReranker`` scores (query, passage) pairs jointly with a BGE cross-encoder
  and returns calibrated probabilities (sigmoid of the logit).
* ``LexicalReranker`` is an offline fallback: IDF-weighted query-term coverage. It is used
  in MOCK mode when the BGE model is not cached, when ``USE_CROSS_ENCODER=false``, or if the
  model fails to load.

When several queries are given (the fallback path), each passage gets its best score across them.
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from typing import Protocol

from app.config import Settings
from app.ingestion.chunker import Chunk
from app.retrieval.hybrid import RetrievedChunk
from app.retrieval.text import tokenize

logger = logging.getLogger(__name__)


class Reranker(Protocol):
    name: str
    default_threshold: float

    def score(self, query: str, chunks: list[Chunk]) -> list[float]: ...


def rerank(
    reranker: Reranker,
    queries: list[str],
    candidates: list[RetrievedChunk],
    top_k: int,
    max_candidates: int = 30,
) -> list[RetrievedChunk]:
    pool = candidates[:max_candidates]
    if not pool:
        return []
    best = [0.0] * len(pool)
    for query in queries:
        for i, s in enumerate(reranker.score(query, [c.chunk for c in pool])):
            best[i] = max(best[i], s)
    scored = [c.with_rerank(s) for c, s in zip(pool, best)]
    scored.sort(key=lambda c: (c.rerank_score, c.fused_score), reverse=True)
    return scored[:top_k]


class LexicalReranker:
    """IDF-weighted query coverage + term-frequency saturation + heading match, in [0, 1].

    Terms that never occur in the corpus get an inflated weight: a question about something
    the policies never mention ("capital of France") should not pass on one incidental word.
    """

    name = "lexical-idf"
    default_threshold = 0.35

    def __init__(self, corpus: list[Chunk]):
        df: Counter = Counter()
        for c in corpus:
            df.update(set(tokenize(self._text(c))))
        n = len(corpus)
        self._idf = {t: math.log((n + 1) / (f + 0.5)) for t, f in df.items()}
        self._unseen_idf = 1.5 * math.log((n + 1) / 0.5)

    @staticmethod
    def _text(chunk: Chunk) -> str:
        # Exclude the document title: it is shared by every section and would inflate coverage.
        return f"{chunk.breadcrumb}\n{chunk.text}"

    def _idf_of(self, token: str) -> float:
        return self._idf.get(token, self._unseen_idf)

    def score(self, query: str, chunks: list[Chunk]) -> list[float]:
        q_terms = set(tokenize(query))
        if not q_terms:
            return [0.0] * len(chunks)
        total = sum(self._idf_of(t) for t in q_terms)
        scores = []
        for chunk in chunks:
            tf = Counter(tokenize(self._text(chunk)))
            heading = set(tokenize(chunk.breadcrumb))
            matched = [t for t in q_terms if tf[t]]
            coverage = sum(self._idf_of(t) for t in matched) / total
            saturation = sum(self._idf_of(t) * tf[t] / (tf[t] + 1.2) for t in matched) / total
            heading_hit = sum(self._idf_of(t) for t in q_terms & heading) / total
            score = 0.7 * coverage + 0.15 * saturation / (3 / 4.2) + 0.15 * heading_hit
            scores.append(round(min(1.0, score), 4))
        return scores


class CrossEncoderReranker:
    default_threshold = 0.20

    def __init__(self, model_name: str, local_files_only: bool = False, max_length: int = 512):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, local_files_only=local_files_only
        )
        self.model.eval()
        self.max_length = max_length
        self.name = model_name

    def score(self, query: str, chunks: list[Chunk], batch_size: int = 16) -> list[float]:
        scores: list[float] = []
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            pairs = [[query, c.index_text] for c in batch]
            with self._torch.no_grad():
                inputs = self.tokenizer(
                    pairs, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
                )
                logits = self.model(**inputs).logits.view(-1).float()
                scores.extend(self._torch.sigmoid(logits).tolist())
        return [round(s, 4) for s in scores]


def get_reranker(settings: Settings, corpus: list[Chunk]) -> Reranker:
    if settings.use_cross_encoder:
        try:
            # In mock mode never trigger a download: only use the model if it is already cached.
            return CrossEncoderReranker(settings.reranker_model, local_files_only=settings.is_mock)
        except Exception as exc:  # missing torch/transformers, no cache, no network, ...
            logger.warning(
                "Cross-encoder %s unavailable (%s); using lexical reranker. "
                "Run scripts/download_models.py to cache it for offline use.",
                settings.reranker_model, exc.__class__.__name__,
            )
    return LexicalReranker(corpus)
