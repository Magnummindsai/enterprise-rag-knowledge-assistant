"""Lightweight text normalisation shared by BM25, the hashing embedder and mock components."""

from __future__ import annotations

import re

_TOKEN = re.compile(r"[a-z0-9]+")

STOPWORDS = frozenset(
    """a an the and or but if of to in on at by for from with as is are was were be been being
    do does did doing have has had i me my we our you your he she they them their it its this that
    these those what which who whom when where why how can could should would may might must will
    shall am any all some there here about into over than then so such not no nor only own same too
    very just also get got many much per please tell know need want allowed allow""".split()
)


def _stem(word: str) -> str:
    """A deliberately small suffix stripper; good enough to align plurals and verb forms."""
    if len(word) <= 3 or word.isdigit():
        return word
    if word.endswith("sses"):
        return word[:-2]
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    for suffix in ("ing", "ed"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            word = word[: -len(suffix)]
            if len(word) > 3 and word[-1] == word[-2] and word[-1] not in "aeiou":
                word = word[:-1]  # travelling -> travel, planned -> plan
            return word
    if word.endswith("s") and not word.endswith(("ss", "us", "is")):
        word = word[:-1]
    if word.endswith("e") and len(word) > 4:
        word = word[:-1]
    return word


def tokenize(text: str, remove_stopwords: bool = True) -> list[str]:
    tokens = _TOKEN.findall(text.lower())
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return [_stem(t) for t in tokens]
