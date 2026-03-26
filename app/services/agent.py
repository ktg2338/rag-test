"""LangGraph 기반 Agentic RAG 워크플로우.

그래프 구조:
    START → route_query →[retrieve_documents | direct_answer]
    retrieve_documents → grade_documents →[generate | rewrite_query]
    rewrite_query → retrieve_documents  (루프)
    generate → hallucination_check →[end | rewrite_query]
    direct_answer → END
"""

from __future__ import annotations

import logging

from langgraph.graph import END, StateGraph

from app.services.graph_state import AgentState
from app.services.nodes import (
    decide_after_grading,
    decide_after_hallucination,
    decide_route,
    direct_answer,
    generate,
    grade_documents,
    hallucination_check,
    retrieve_documents,
    rewrite_query,
    route_query,
)

logger = logging.getLogger(__name__)


def build_graph() -> StateGraph:
    """Agentic RAG 그래프를 구성하고 컴파일한다."""
    graph = StateGraph(AgentState)

    # 노드 등록
    graph.add_node("route_query", route_query)
    graph.add_node("retrieve_documents", retrieve_documents)
    graph.add_node("grade_documents", grade_documents)
    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("generate", generate)
    graph.add_node("direct_answer", direct_answer)
    graph.add_node("hallucination_check", hallucination_check)

    # 엣지 연결
    graph.set_entry_point("route_query")

    # route_query → retrieve_documents 또는 direct_answer
    graph.add_conditional_edges(
        "route_query",
        decide_route,
        {
            "retrieve_documents": "retrieve_documents",
            "direct_answer": "direct_answer",
        },
    )

    # retrieve_documents → grade_documents
    graph.add_edge("retrieve_documents", "grade_documents")

    # grade_documents → generate 또는 rewrite_query
    graph.add_conditional_edges(
        "grade_documents",
        decide_after_grading,
        {
            "generate": "generate",
            "rewrite_query": "rewrite_query",
        },
    )

    # rewrite_query → retrieve_documents (루프)
    graph.add_edge("rewrite_query", "retrieve_documents")

    # generate → hallucination_check
    graph.add_edge("generate", "hallucination_check")

    # hallucination_check → end 또는 rewrite_query
    graph.add_conditional_edges(
        "hallucination_check",
        decide_after_hallucination,
        {
            "end": END,
            "rewrite_query": "rewrite_query",
        },
    )

    # direct_answer → END
    graph.add_edge("direct_answer", END)

    return graph.compile()


# 싱글톤 컴파일된 그래프
rag_agent = build_graph()
