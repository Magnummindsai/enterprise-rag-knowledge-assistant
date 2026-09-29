"""Application settings, loaded from environment variables and an optional .env file."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM / embeddings ---
    openai_api_key: str = ""
    mock_mode: bool = False
    openai_chat_model: str = "gpt-4o"
    openai_embedding_model: str = "text-embedding-3-small"

    # --- Reranking ---
    reranker_model: str = "BAAI/bge-reranker-base"
    use_cross_encoder: bool = True
    relevance_threshold: float | None = None  # None -> reranker-specific default

    # --- Retrieval ---
    top_k_retrieve: int = 20
    top_k_rerank: int = 5

    # --- Paths & persistence ---
    data_dir: Path = BASE_DIR / "data" / "policies"
    index_dir: Path = BASE_DIR / "storage" / "index"
    database_url: str = f"sqlite:///{(BASE_DIR / 'storage' / 'rag.db').as_posix()}"

    log_level: str = "INFO"

    @property
    def is_mock(self) -> bool:
        """Mock mode is used when forced, or automatically when no API key is configured."""
        return self.mock_mode or not self.openai_api_key.strip()


@lru_cache
def get_settings() -> Settings:
    return Settings()
