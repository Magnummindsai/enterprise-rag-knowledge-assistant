"""OpenAI chat model (GPT-4o by default) for grounded answers and query rewriting."""

from __future__ import annotations

import json
import logging

from openai import OpenAI

from app.llm.prompts import build_answer_messages, build_rewrite_messages
from app.retrieval.hybrid import RetrievedChunk

logger = logging.getLogger(__name__)


class OpenAILLM:
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self.client = OpenAI(api_key=api_key, timeout=45, max_retries=2)
        self.model = model
        self.name = model

    def generate_answer(self, question: str, contexts: list[RetrievedChunk]) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=build_answer_messages(question, contexts),
            temperature=0.0,
            max_tokens=600,
        )
        return (response.choices[0].message.content or "").strip()

    def rewrite_queries(self, question: str) -> list[str]:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=build_rewrite_messages(question),
            temperature=0.2,
            max_tokens=200,
            response_format={"type": "json_object"},
        )
        try:
            queries = json.loads(response.choices[0].message.content or "{}").get("queries", [])
        except json.JSONDecodeError:
            logger.warning("Query rewrite returned invalid JSON")
            return []
        return [q.strip() for q in queries if isinstance(q, str) and q.strip()][:3]
