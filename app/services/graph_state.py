"""Agentic RAG 그래프의 상태 정의."""

from __future__ import annotations

from typing import List, TypedDict

from app.services.memory import Message


class AgentState(TypedDict, total=False):
    """LangGraph 워크플로우에서 노드 간 전달되는 상태."""

    # 입력
    question: str
    conversation_id: str
    top_k: int
    history: List[Message]

    # 중간 상태
    documents: List[str]          # 검색된 문서들
    filtered_documents: List[str] # 평가를 통과한 문서들
    rewritten_query: str          # 재작성된 쿼리
    retry_count: int              # 재시도 횟수

    # 출력
    generation: str               # 최종 답변
    steps: List[str]              # Agent가 수행한 단계 기록
