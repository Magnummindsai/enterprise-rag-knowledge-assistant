"""Tests run fully offline: MOCK mode, lexical reranker, temporary index and database."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

_TMP = Path(tempfile.mkdtemp(prefix="rag-tests-"))
# Must be set before app modules read settings.
os.environ.update(
    {
        "MOCK_MODE": "true",
        "OPENAI_API_KEY": "",
        "USE_CROSS_ENCODER": "false",
        "INDEX_DIR": str(_TMP / "index"),
        "DATABASE_URL": f"sqlite:///{(_TMP / 'test.db').as_posix()}",
    }
)

from app.config import Settings, get_settings  # noqa: E402
from app.pipeline import RAGPipeline  # noqa: E402


@pytest.fixture(scope="session")
def settings() -> Settings:
    get_settings.cache_clear()
    return get_settings()


@pytest.fixture(scope="session")
def pipeline(settings: Settings) -> RAGPipeline:
    return RAGPipeline(settings)
