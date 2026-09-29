"""Offline stand-in for the LLM, used in MOCK mode and as a degraded fallback.

* Query rewriting uses a small synonym table mapping everyday phrasing to policy vocabulary.
* Answers are *extractive*: the most relevant sentences from the top reranked sections,
  each group followed by its citation. Nothing is generated, so nothing can be hallucinated.
"""

from __future__ import annotations

import re

from app.ingestion.chunker import split_sentences
from app.llm.prompts import NOT_FOUND_ANSWER
from app.retrieval.hybrid import RetrievedChunk
from app.retrieval.text import tokenize

SYNONYMS: dict[str, str] = {
    "pto": "annual leave entitlement",
    "vacation": "annual leave",
    "holiday": "annual leave",
    "holidays": "annual leave",
    "days off": "annual leave",
    "time off": "leave",
    "roll over": "carry-over unused annual leave",
    "rollover": "carry-over unused annual leave",
    "sick day": "sick leave",
    "sick days": "sick leave",
    "doctor's note": "medical certificate",
    "doctors note": "medical certificate",
    "maternity": "birth parent parental leave",
    "paternity": "non-birth parent parental leave",
    "baby": "parental leave",
    "adopt": "adoption leave",
    "wfh": "remote work",
    "work from home": "remote work home",
    "working from home": "remote work home",
    "abroad": "working from another country",
    "desk": "home office allowance",
    "chair": "home office allowance",
    "monitor": "equipment",
    "laptop": "device equipment",
    "per diem": "meal per diem",
    "food": "meal per diem",
    "meals": "meal per diem",
    "hotel": "accommodation hotel rate cap",
    "fly": "air travel class of travel",
    "flight": "air travel class of travel",
    "business class": "air travel class of travel",
    "uber": "taxis ride-hailing",
    "taxi": "taxis ride-hailing",
    "mileage": "personal vehicles per mile",
    "reimburse": "expense claim reimbursement",
    "reimbursed": "expense claim reimbursement",
    "receipt": "receipts expense claims",
    "lost receipt": "missing receipt declaration",
    "credit card": "corporate card",
    "client dinner": "client entertainment",
    "beer": "alcohol",
    "wine": "alcohol",
    "2fa": "multi-factor authentication mfa",
    "two factor": "multi-factor authentication mfa",
    "passcode": "password requirements",
    "phishing": "report phishing suspicious email",
    "hacked": "security incident account compromise",
    "virus": "security incident malware",
    "stolen": "lost stolen device security incident",
    "chatgpt": "generative ai tools",
    "ai tool": "generative ai tools",
    "usb": "removable media",
    "buy": "purchase procurement",
    "vendor": "supplier",
    "quotes": "competitive quotes",
    "po": "purchase order",
    "sign a contract": "signing authority contracts",
}


def expand_query(question: str) -> list[str]:
    q = question.lower()
    expansions = [
        target for phrase, target in SYNONYMS.items() if re.search(rf"\b{re.escape(phrase)}\b", q)
    ]
    return list(dict.fromkeys(expansions))


class MockLLM:
    name = "mock-extractive"

    def rewrite_queries(self, question: str) -> list[str]:
        expansions = expand_query(question)
        keywords = " ".join(tokenize(question, remove_stopwords=True))
        candidates = [
            f"{question} {' '.join(expansions)}".strip(),
            f"{keywords} {' '.join(expansions)}".strip(),
            " ".join(expansions),
        ]
        seen, queries = set(), []
        for c in candidates:
            if c and c.lower() not in seen and c.lower() != question.lower():
                seen.add(c.lower())
                queries.append(c)
        return queries[:3]

    def generate_answer(self, question: str, contexts: list[RetrievedChunk]) -> str:
        if not contexts:
            return NOT_FOUND_ANSWER
        q_terms = set(tokenize(question)) | set(tokenize(" ".join(expand_query(question))))

        wants_quantity = bool(re.search(r"\b(how (many|much|long|often|soon)|what is the|limit|cap|maximum|minimum)\b", question.lower()))

        scored = []  # (score, ctx_rank, sent_pos, sentence, chunk_id)
        for rank, rc in enumerate(contexts[:3]):
            heading_terms = set(tokenize(rc.chunk.breadcrumb))
            for pos, sentence in enumerate(split_sentences(rc.chunk.text)):
                terms = set(tokenize(sentence))
                overlap = len(q_terms & terms) + 0.5 * len(q_terms & heading_terms)
                if not q_terms & terms:
                    continue
                if wants_quantity and re.search(r"\d", sentence):
                    overlap += 1.0  # questions about limits are answered by sentences with numbers
                scored.append((overlap - 0.75 * rank, rank, pos, sentence, rc.chunk.chunk_id))

        if not scored:  # fall back to the opening of the best section
            top = contexts[0].chunk
            lead = " ".join(split_sentences(top.text)[:2])
            return f"{lead} [{top.chunk_id}]"

        best = sorted(scored, key=lambda s: s[0], reverse=True)[:3]
        best.sort(key=lambda s: (s[1], s[2]))  # keep document order for readability

        parts: list[str] = []
        current_id, buffer = None, []
        for _, _, _, sentence, chunk_id in best:
            if chunk_id != current_id and buffer:
                parts.append(f"{' '.join(buffer)} [{current_id}]")
                buffer = []
            current_id = chunk_id
            buffer.append(sentence)
        parts.append(f"{' '.join(buffer)} [{current_id}]")
        return " ".join(parts)
