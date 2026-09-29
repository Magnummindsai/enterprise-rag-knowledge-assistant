"""Assemble the LangGraph state machine."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from app.graph.nodes import GraphNodes
from app.graph.state import GraphState


def build_graph(nodes: GraphNodes):
    graph = StateGraph(GraphState)
    graph.add_node("analyze_query", nodes.analyze_query)
    graph.add_node("hybrid_retrieve", nodes.hybrid_retrieve)
    graph.add_node("rerank", nodes.rerank)
    graph.add_node("fallback_retrieve", nodes.fallback_retrieve)
    graph.add_node("generate", nodes.generate)
    graph.add_node("validate_citations", nodes.validate_citations)
    graph.add_node("abstain", nodes.abstain)

    graph.add_edge(START, "analyze_query")
    graph.add_edge("analyze_query", "hybrid_retrieve")
    graph.add_edge("hybrid_retrieve", "rerank")
    graph.add_conditional_edges(
        "rerank",
        nodes.route_after_rerank,
        {"generate": "generate", "fallback": "fallback_retrieve", "abstain": "abstain"},
    )
    graph.add_edge("fallback_retrieve", "rerank")
    graph.add_edge("generate", "validate_citations")
    graph.add_edge("validate_citations", END)
    graph.add_edge("abstain", END)
    return graph.compile()
