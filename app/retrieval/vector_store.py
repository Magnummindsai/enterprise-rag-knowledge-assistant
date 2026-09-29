"""Dense retrieval with FAISS, persisted to disk and rebuilt automatically when the corpus changes."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path

import faiss
import numpy as np

from app.ingestion.chunker import Chunk
from app.retrieval.embeddings import Embedder

logger = logging.getLogger(__name__)


def corpus_fingerprint(chunks: list[Chunk]) -> str:
    digest = hashlib.sha256()
    for c in chunks:
        digest.update(c.chunk_id.encode())
        digest.update(c.index_text.encode())
    return digest.hexdigest()[:16]


class VectorStore:
    """Exact inner-product search over L2-normalised vectors (i.e. cosine similarity).

    ``IndexFlatIP`` is the right choice for a corpus of this size; swap in ``IndexHNSWFlat``
    or ``IndexIVFFlat`` for millions of chunks.
    """

    def __init__(self, embeddings: np.ndarray, embedder: Embedder):
        self.embedder = embedder
        self.embeddings = embeddings.astype(np.float32)
        self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
        self.index.add(self.embeddings)

    def __len__(self) -> int:
        return self.index.ntotal

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        query_vec = self.embedder.embed([query])
        scores, ids = self.index.search(query_vec, min(k, len(self)))
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0]

    @classmethod
    def load_or_build(cls, chunks: list[Chunk], embedder: Embedder, index_dir: Path) -> VectorStore:
        """Embeddings are cached per embedder; a corpus fingerprint triggers a rebuild on change.

        Embeddings are stored as .npy (rather than a serialised FAISS index) so the cache is
        portable and the flat index is rebuilt in milliseconds on load.
        """
        target = Path(index_dir) / re.sub(r"[^a-zA-Z0-9_.-]", "_", embedder.name)
        manifest_path, vectors_path = target / "manifest.json", target / "embeddings.npy"
        fingerprint = corpus_fingerprint(chunks)

        if manifest_path.exists() and vectors_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if manifest.get("fingerprint") == fingerprint:
                logger.info("Loaded cached embeddings from %s", target)
                return cls(np.load(vectors_path), embedder)
            logger.info("Corpus changed; rebuilding embeddings")

        logger.info("Embedding %d chunks with %s", len(chunks), embedder.name)
        embeddings = embedder.embed([c.index_text for c in chunks])
        target.mkdir(parents=True, exist_ok=True)
        np.save(vectors_path, embeddings)
        manifest_path.write_text(
            json.dumps(
                {
                    "fingerprint": fingerprint,
                    "embedder": embedder.name,
                    "num_chunks": len(chunks),
                    "dim": int(embeddings.shape[1]),
                    "chunk_ids": [c.chunk_id for c in chunks],
                },
                indent=2,
            )
        )
        return cls(embeddings, embedder)
