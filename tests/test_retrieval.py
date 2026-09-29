import numpy as np
import pytest

from app.retrieval.embeddings import HashingEmbedder
from app.retrieval.hybrid import reciprocal_rank_fusion
from app.retrieval.reranker import LexicalReranker, rerank
from app.retrieval.text import tokenize


def test_tokenizer_normalises_word_forms():
    assert tokenize("Travelling expenses") == tokenize("travel expense")
    assert "the" not in tokenize("What is the policy?")


def test_hashing_embedder_is_deterministic_and_normalised():
    emb = HashingEmbedder(dim=256)
    a, b = emb.embed(["hotel rate cap", "hotel rate cap"])
    assert np.allclose(a, b)
    assert np.isclose(np.linalg.norm(a), 1.0, atol=1e-5)


def test_rrf_rewards_agreement_between_retrievers():
    fused = reciprocal_rank_fusion([([1, 2, 3], 1.0), ([2, 1, 4], 1.0)])
    assert fused[1] == pytest.approx(fused[2])
    assert fused[1] > fused[3] > 0
    assert 4 in fused


def test_rrf_weights_shift_ranking():
    fused = reciprocal_rank_fusion([([1], 2.0), ([2], 1.0)])
    assert fused[1] > fused[2]


def test_bm25_finds_exact_terms(pipeline):
    hits = pipeline.retriever.bm25.search("hotel nightly caps Tier 1 cities", k=3)
    assert pipeline.chunks[hits[0][0]].chunk_id == "TRV-5.1"


def test_vector_store_returns_k_results(pipeline):
    hits = pipeline.retriever.vectors.search("corporate card monthly limit", k=5)
    assert len(hits) == 5
    assert "EXP-4.1" in [pipeline.chunks[i].chunk_id for i, _ in hits]


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("What is the hotel rate cap in London?", "TRV-5.1"),
        ("per diem for international travel", "TRV-6.1"),
        ("How long do I have to submit an expense claim?", "EXP-2.2"),
        ("minimum password length", "PAC-2.1"),
        ("how many quotes do I need for a purchase", "PRC-2.1"),
        ("report a phishing email", "ITS-5.1"),
    ],
)
def test_hybrid_retrieval_recall_at_5(pipeline, query, expected):
    results = pipeline.retriever.search(query, k=5)
    assert expected in [r.chunk.chunk_id for r in results]
    assert all(r.bm25_rank or r.vector_rank for r in results)


def test_lexical_reranker_prefers_relevant_section(pipeline):
    reranker = LexicalReranker(pipeline.chunks)
    candidates = pipeline.retriever.search("business class flights over 10 hours", k=20)
    top = rerank(reranker, ["business class flights over 10 hours"], candidates, top_k=3)
    assert top[0].chunk.chunk_id == "TRV-3.1"
    assert top[0].rerank_score >= top[-1].rerank_score


def test_lexical_reranker_scores_off_topic_low(pipeline):
    reranker = LexicalReranker(pipeline.chunks)
    scores = reranker.score("Who won the football world cup?", pipeline.chunks[:30])
    assert max(scores) < reranker.default_threshold
