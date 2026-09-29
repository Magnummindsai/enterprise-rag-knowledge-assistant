"""Embedding backends: OpenAI (real mode) and a deterministic hashing embedder (mock mode)."""

from __future__ import annotations

import logging
import math
import zlib
from collections import Counter
from typing import Protocol

import numpy as np

from app.config import Settings
from app.retrieval.text import tokenize

logger = logging.getLogger(__name__)


class Embedder(Protocol):
    name: str

    def embed(self, texts: list[str]) -> np.ndarray: ...


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


class HashingEmbedder:
    """Feature-hashing embedder over word unigrams, bigrams and character trigrams.

    Fully offline and deterministic, so MOCK mode and tests need no network or model
    download. It is a lexical approximation of semantic search, not a replacement for it.
    """

    def __init__(self, dim: int = 1024):
        self.dim = dim
        self.name = f"hashing-{dim}"

    def _features(self, text: str) -> Counter:
        tokens = tokenize(text)
        feats: Counter = Counter()
        feats.update(f"w:{t}" for t in tokens)
        feats.update(f"b:{a}_{b}" for a, b in zip(tokens, tokens[1:]))
        for t in tokens:
            padded = f"#{t}#"
            for i in range(len(padded) - 2):
                feats[f"c:{padded[i:i + 3]}"] += 0.3
        return feats

    def embed(self, texts: list[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for feat, count in self._features(text).items():
                h = zlib.crc32(feat.encode("utf-8"))
                sign = 1.0 if (h >> 31) & 1 else -1.0
                matrix[row, h % self.dim] += sign * math.log1p(count)  # sublinear tf
        return _l2_normalize(matrix)


class OpenAIEmbedder:
    def __init__(self, api_key: str, model: str, batch_size: int = 96):
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key, timeout=60)
        self.model = model
        self.batch_size = batch_size
        self.name = f"openai-{model}"

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = [t.replace("\n", " ") for t in texts[start : start + self.batch_size]]
            response = self.client.embeddings.create(model=self.model, input=batch)
            vectors.extend(item.embedding for item in response.data)
        return _l2_normalize(np.asarray(vectors, dtype=np.float32))


def get_embedder(settings: Settings) -> Embedder:
    if settings.is_mock:
        return HashingEmbedder()
    return OpenAIEmbedder(settings.openai_api_key, settings.openai_embedding_model)
