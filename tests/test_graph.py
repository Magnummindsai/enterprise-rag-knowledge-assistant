from app.graph.nodes import ABSTAIN_MESSAGE, CITATION_PATTERN, validate_citations
from app.llm.mock_llm import MockLLM, expand_query
from app.retrieval.hybrid import RetrievedChunk


def test_direct_question_is_answered_with_citations(pipeline):
    result = pipeline.ask("What is the hotel rate cap in London?")
    assert not result.used_fallback
    assert result.confidence == "high"
    assert "USD 250" in result.answer
    assert "TRV-5.1" in [c["id"] for c in result.citations]


def test_every_citation_in_answer_is_backed_by_context(pipeline):
    result = pipeline.ask("How long do I have to submit an expense claim?")
    cited = set(CITATION_PATTERN.findall(result.answer))
    assert cited, "answer should carry citations"
    assert cited <= set(result.context_ids)
    assert cited == {c["id"] for c in result.citations}


def test_vocabulary_gap_triggers_fallback_path(pipeline):
    # "PTO" / "roll over" never appear in the policies; the fallback rewrites the query.
    result = pipeline.ask("How much PTO can I roll over?")
    assert result.used_fallback
    assert len(result.queries) > 1
    assert "LEV-2.4" in [c["id"] for c in result.citations]
    assert result.confidence == "medium"
    nodes = [t["node"] for t in result.trace]
    assert nodes.count("rerank") == 2 and "fallback_retrieve" in nodes


def test_out_of_scope_question_abstains(pipeline):
    result = pipeline.ask("Who won the football world cup?")
    assert result.answer == ABSTAIN_MESSAGE
    assert result.citations == []
    assert result.confidence == "low"
    assert result.used_fallback  # it tried the fallback before giving up
    assert [t["node"] for t in result.trace][-1] == "abstain"


def test_trace_records_the_graph_path(pipeline):
    result = pipeline.ask("minimum password length")
    assert [t["node"] for t in result.trace][:3] == ["analyze_query", "hybrid_retrieve", "rerank"]
    assert all(t["ms"] >= 0 for t in result.trace)


def test_validator_strips_hallucinated_citations(pipeline):
    ctx = [RetrievedChunk(chunk=pipeline.get_chunk("TRV-5.1"), fused_score=1.0, rerank_score=0.9)]
    answer, citations, invalid = validate_citations("The cap is USD 250 [TRV-5.1][XYZ-9.9].", ctx)
    assert answer == "The cap is USD 250 [TRV-5.1]."
    assert [c["id"] for c in citations] == ["TRV-5.1"]
    assert invalid == ["XYZ-9.9"]


def test_llm_failure_degrades_to_extractive_answer(pipeline, monkeypatch):
    def boom(*_args, **_kwargs):
        raise RuntimeError("provider down")

    monkeypatch.setattr(pipeline.llm, "generate_answer", boom)
    result = pipeline.ask("What is the hotel rate cap in London?")
    assert "llm_unavailable_extractive_answer" in result.warnings
    assert result.citations


def test_mock_query_rewriting_maps_to_policy_vocabulary():
    assert "remote work" in " ".join(expand_query("Can I WFH on Fridays?"))
    queries = MockLLM().rewrite_queries("How many sick days do I get?")
    assert 1 <= len(queries) <= 3
    assert any("sick leave" in q for q in queries)
