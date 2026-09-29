"""Section-aware chunking.

Policy documents are split on their numbered headings so each chunk maps to a citable
section (e.g. ``TRV-5.1``). Sections longer than ``max_chars`` are split on sentence
boundaries into parts (``TRV-5.1.p2``).
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from app.ingestion.loader import PolicyDocument

_H2 = re.compile(r"^##\s+(\d+)\.?\s+(.*)$")
_H3 = re.compile(r"^###\s+(\d+(?:\.\d+)*)\.?\s+(.*)$")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"(])")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    doc_title: str
    section: str
    heading: str
    parent_heading: str
    text: str
    source: str

    @property
    def breadcrumb(self) -> str:
        if self.parent_heading and self.parent_heading != self.heading:
            return f"{self.parent_heading} > {self.heading}"
        return self.heading

    @property
    def index_text(self) -> str:
        """Text used for BM25 / embeddings: headings add useful retrieval signal."""
        return f"{self.doc_title}\n{self.breadcrumb}\n{self.text}"

    def to_dict(self) -> dict:
        return asdict(self)


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]


def _split_long(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    parts: list[str] = []
    current = ""
    for sentence in split_sentences(text):
        if current and len(current) + len(sentence) + 1 > max_chars:
            parts.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        parts.append(current)
    return parts


def chunk_document(doc: PolicyDocument, max_chars: int = 1200) -> list[Chunk]:
    chunks: list[Chunk] = []
    h2_num, h2_title = "", ""
    section, heading = "", ""
    buffer: list[str] = []

    def flush() -> None:
        text = " ".join(line.strip() for line in buffer if line.strip())
        buffer.clear()
        if not text or not section:
            return
        for i, part in enumerate(_split_long(text, max_chars), start=1):
            suffix = "" if i == 1 else f".p{i}"
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}-{section}{suffix}",
                    doc_id=doc.doc_id,
                    doc_title=doc.title,
                    section=section,
                    heading=heading,
                    parent_heading=h2_title,
                    text=part,
                    source=doc.source,
                )
            )

    for line in doc.body.splitlines():
        stripped = line.strip()
        if stripped.startswith("# ") or stripped.startswith(">"):
            continue  # document title / disclaimer banner
        if m := _H2.match(stripped):
            flush()
            h2_num, h2_title = m.group(1), m.group(2).strip()
            section, heading = h2_num, h2_title
        elif m := _H3.match(stripped):
            flush()
            section, heading = m.group(1), m.group(2).strip()
        else:
            buffer.append(stripped)
    flush()
    return chunks


def chunk_documents(docs: list[PolicyDocument], max_chars: int = 1200) -> list[Chunk]:
    chunks = [c for doc in docs for c in chunk_document(doc, max_chars)]
    ids = [c.chunk_id for c in chunks]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise ValueError(f"Duplicate chunk ids: {sorted(dupes)}")
    return chunks
