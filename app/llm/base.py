"""LLM interface and factory."""

from __future__ import annotations

from typing import Protocol

from app.config import Settings
from app.retrieval.hybrid import RetrievedChunk


class LLM(Protocol):
    name: str

    def generate_answer(self, question: str, contexts: list[RetrievedChunk]) -> str: ...

    def rewrite_queries(self, question: str) -> list[str]: ...


def get_llm(settings: Settings) -> LLM:
    if settings.is_mock:
        from app.llm.mock_llm import MockLLM

        return MockLLM()
    from app.llm.openai_client import OpenAILLM

    return OpenAILLM(settings.openai_api_key, settings.openai_chat_model)
