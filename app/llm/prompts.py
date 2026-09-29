"""Prompt templates for answer generation and query rewriting."""

from __future__ import annotations

from app.retrieval.hybrid import RetrievedChunk

NOT_FOUND_ANSWER = "I couldn't find this in the policy documents."

ANSWER_SYSTEM_PROMPT = f"""You are the Northwind Dynamics Policy Assistant. You answer employee \
questions using ONLY the policy excerpts provided in the user message.

Rules:
1. Every factual sentence must end with one or more citations using the exact excerpt IDs in \
square brackets, for example [TRV-5.1] or [EXP-2.3][TRV-6.1].
2. Only cite IDs that appear in the provided excerpts. Never invent IDs, numbers or rules.
3. If the excerpts do not contain the answer, reply with exactly: "{NOT_FOUND_ANSWER}"
4. If the excerpts answer only part of the question, answer that part and state clearly what \
the policies do not cover.
5. Be concise: 1-4 sentences or a short bulleted list. Quote exact numbers, limits, deadlines \
and approvers.
6. Do not give legal or tax advice. For unusual situations, suggest contacting the policy owner."""


def format_context(contexts: list[RetrievedChunk]) -> str:
    blocks = []
    for rc in contexts:
        c = rc.chunk
        blocks.append(f"[{c.chunk_id}] {c.doc_title} - {c.section} {c.breadcrumb}\n{c.text}")
    return "\n\n".join(blocks)


def build_answer_messages(question: str, contexts: list[RetrievedChunk]) -> list[dict]:
    user = f"Policy excerpts:\n\n{format_context(contexts)}\n\nEmployee question: {question}"
    return [
        {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


REWRITE_SYSTEM_PROMPT = """You help a search engine find passages in internal company policy \
documents (leave, parental leave, travel, expenses, IT security, passwords, procurement, remote work).

Rewrite the employee's question into up to 3 alternative search queries that use the formal \
vocabulary a policy document would use (e.g. "PTO" -> "annual leave entitlement", "WFH" -> \
"remote work"). Keep each query short. Respond with JSON: {"queries": ["...", "..."]}"""


def build_rewrite_messages(question: str) -> list[dict]:
    return [
        {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
