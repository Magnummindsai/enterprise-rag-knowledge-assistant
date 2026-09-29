import re

import pytest

from app.ingestion.chunker import chunk_document, chunk_documents
from app.ingestion.loader import PolicyDocument, load_documents


def test_loads_all_policy_documents(settings):
    docs = load_documents(settings.data_dir)
    assert len(docs) == 8
    assert {d.doc_id for d in docs} == {"LEV", "PFL", "TRV", "EXP", "ITS", "PAC", "PRC", "RWE"}
    assert all(d.title and d.metadata.get("owner") for d in docs)


def test_chunk_ids_are_unique_and_citable(settings):
    chunks = chunk_documents(load_documents(settings.data_dir))
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
    assert all(re.fullmatch(r"[A-Z]{2,5}-\d+(\.\d+)*(\.p\d+)?", i) for i in ids)


def test_section_content_is_preserved(settings):
    chunks = {c.chunk_id: c for c in chunk_documents(load_documents(settings.data_dir))}
    hotel = chunks["TRV-5.1"]
    assert hotel.heading == "Hotel Rate Caps"
    assert hotel.parent_heading == "Accommodation"
    assert "USD 250" in hotel.text
    assert "Synthetic document" not in hotel.text  # disclaimer banner is skipped


def test_sections_without_subsections_are_chunked(settings):
    chunks = {c.chunk_id for c in chunk_documents(load_documents(settings.data_dir))}
    assert "PAC-1" in chunks  # "## 1. Purpose" has body text directly under it


def test_long_sections_are_split_into_parts():
    body = "## 1. Scope\n### 1.1 Long\n" + " ".join(f"Sentence number {i} is here." for i in range(200))
    doc = PolicyDocument(doc_id="TST", title="Test", body=body, source="t.md")
    chunks = chunk_document(doc, max_chars=500)
    assert len(chunks) > 1
    assert chunks[0].chunk_id == "TST-1.1" and chunks[1].chunk_id == "TST-1.1.p2"
    assert all(len(c.text) <= 500 for c in chunks)


def test_missing_front_matter_is_rejected(tmp_path):
    (tmp_path / "bad.md").write_text("# No front matter\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_documents(tmp_path)
