"""Load Markdown policy documents with a simple YAML-style front-matter header."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class PolicyDocument:
    doc_id: str
    title: str
    body: str
    source: str
    metadata: dict[str, str] = field(default_factory=dict)


def _parse_front_matter(raw: str) -> tuple[dict[str, str], str]:
    """Split `---`-delimited front matter from the body. Values are kept as strings."""
    lines = raw.lstrip("﻿").splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, raw
    meta: dict[str, str] = {}
    for i, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            return meta, "\n".join(lines[i + 1 :])
        if ":" in line:
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip()
    raise ValueError("Unterminated front matter block")


def load_document(path: Path) -> PolicyDocument:
    meta, body = _parse_front_matter(path.read_text(encoding="utf-8"))
    missing = {"doc_id", "title"} - meta.keys()
    if missing:
        raise ValueError(f"{path.name}: missing front matter keys {sorted(missing)}")
    return PolicyDocument(
        doc_id=meta["doc_id"].upper(),
        title=meta["title"],
        body=body.strip(),
        source=path.name,
        metadata=meta,
    )


def load_documents(data_dir: Path) -> list[PolicyDocument]:
    paths = sorted(Path(data_dir).glob("*.md"))
    if not paths:
        raise FileNotFoundError(f"No policy documents (*.md) found in {data_dir}")
    docs = [load_document(p) for p in paths]
    ids = [d.doc_id for d in docs]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate doc_id values: {ids}")
    return docs
