"""Agentic RAG 그래프의 노드 함수들.

각 노드는 AgentState를 받아 상태 업데이트 dict를 반환한다.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict

from openai import AzureOpenAI

from app.core.config import settings
from app.services.graph_state import AgentState
from app.services.retriever import retrieve

logger = logging.getLogger(__name__)

_client = AzureOpenAI(
    api_key=settings.AZURE_OPENAI_API_KEY,
    azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
    api_version=settings.AZURE_OPENAI_API_VERSION,
)

MAX_RETRIES = 2


# ── 유틸 ──────────────────────────────────────────────

def _llm(system: str, user: str, *, temperature: float = 0.0) -> str:
    """간단한 LLM 호출 래퍼."""
    resp = _client.chat.completions.create(
        model=settings.AZURE_OPENAI_DEPLOYMENT,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
    )
    return resp.choices[0].message.content.strip()


def _add_step(state: AgentState, step: str) -> list[str]:
    """steps 리스트에 단계를 추가하고 반환."""
    steps = list(state.get("steps", []))
    steps.append(step)
    return steps


# ── 노드 함수들 ──────────────────────────────────────

def route_query(state: AgentState) -> Dict[str, Any]:
    """질문을 분석하여 검색이 필요한지 판단한다."""
    question = state.get("rewritten_query") or state["question"]

    result = _llm(
        system=(
            "You are a query router. Decide whether the user question requires "
            "document retrieval or can be answered directly.\n"
            "Respond with ONLY a JSON object: "
            '{\"action\": \"retrieve\"} or {\"action\": \"direct\"}\n'
            "Use \"retrieve\" for factual questions about specific topics. "
            "Use \"direct\" only for greetings or trivial questions like 'hi', 'hello'."
        ),
        user=question,
    )
    try:
        action = json.loads(result).get("action", "retrieve")
    except (json.JSONDecodeError, AttributeError):
        action = "retrieve"

    return {"steps": _add_step(state, f"route_query → {action}")}


def retrieve_documents(state: AgentState) -> Dict[str, Any]:
    """Hybrid Search + Reranking으로 문서를 검색한다."""
    query = state.get("rewritten_query") or state["question"]
    top_k = state.get("top_k", settings.MAX_CONTEXT_CHUNKS)

    docs, _metas = retrieve(query, top_k=top_k)
    logger.info("Retrieved %d documents for query: %s", len(docs), query[:80])

    return {
        "documents": docs,
        "steps": _add_step(state, f"retrieve → {len(docs)}건 검색"),
    }


def grade_documents(state: AgentState) -> Dict[str, Any]:
    """검색된 문서들이 질문에 관련 있는지 LLM으로 평가한다."""
    question = state.get("rewritten_query") or state["question"]
    documents = state.get("documents", [])

    if not documents:
        return {
            "filtered_documents": [],
            "steps": _add_step(state, "grade_documents → 검색 결과 없음"),
        }

    filtered = []
    for i, doc in enumerate(documents):
        result = _llm(
            system=(
                "You are a relevance grader. Given a question and a document, "
                "decide if the document is relevant to answering the question.\n"
                "Respond with ONLY a JSON object: "
                '{\"relevant\": true} or {\"relevant\": false}'
            ),
            user=f"Question: {question}\n\nDocument:\n{doc}",
        )
        try:
            relevant = json.loads(result).get("relevant", True)
        except (json.JSONDecodeError, AttributeError):
            relevant = True  # 파싱 실패 시 포함

        if relevant:
            filtered.append(doc)

    return {
        "filtered_documents": filtered,
        "steps": _add_step(
            state,
            f"grade_documents → {len(filtered)}/{len(documents)}건 관련",
        ),
    }


def rewrite_query(state: AgentState) -> Dict[str, Any]:
    """관련 문서가 부족할 때 쿼리를 재작성한다."""
    question = state["question"]
    retry_count = state.get("retry_count", 0) + 1

    rewritten = _llm(
        system=(
            "You are a query rewriter. Rewrite the user question to improve "
            "document retrieval. Make it more specific or use alternative terms.\n"
            "Return ONLY the rewritten query, nothing else."
        ),
        user=f"Original question: {question}",
    )
    logger.info("Query rewritten: %s → %s", question[:60], rewritten[:60])

    return {
        "rewritten_query": rewritten,
        "retry_count": retry_count,
        "steps": _add_step(state, f"rewrite_query → \"{rewritten[:50]}\""),
    }


def generate(state: AgentState) -> Dict[str, Any]:
    """필터링된 문서를 기반으로 최종 답변을 생성한다."""
    question = state["question"]
    documents = state.get("filtered_documents") or state.get("documents", [])
    history = state.get("history", [])

    context_block = "\n\n---\n\n".join(documents) if documents else "N/A"
    system = (
        "You are a helpful assistant that answers strictly based on the provided context. "
        "If the answer is not contained in the context, say you don't know."
    )
    user_msg = (
        f"# Question\n{question}\n\n"
        f"# Context\n{context_block}\n\n"
        "Answer in Korean. Include brief citations like [#1], [#2] "
        "referring to the order of context chunks if useful."
    )

    messages = [{"role": "system", "content": system}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_msg})

    resp = _client.chat.completions.create(
        model=settings.AZURE_OPENAI_DEPLOYMENT,
        messages=messages,
        temperature=0.2,
    )
    answer = resp.choices[0].message.content.strip()

    return {
        "generation": answer,
        "steps": _add_step(state, "generate → 답변 생성"),
    }


def direct_answer(state: AgentState) -> Dict[str, Any]:
    """검색 없이 바로 답변한다 (인사, 간단한 질문 등)."""
    question = state["question"]
    history = state.get("history", [])

    messages = [
        {"role": "system", "content": "You are a helpful assistant. Answer in Korean."},
    ]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": question})

    resp = _client.chat.completions.create(
        model=settings.AZURE_OPENAI_DEPLOYMENT,
        messages=messages,
        temperature=0.5,
    )
    answer = resp.choices[0].message.content.strip()

    return {
        "generation": answer,
        "documents": [],
        "filtered_documents": [],
        "steps": _add_step(state, "direct_answer → 직접 답변"),
    }


def hallucination_check(state: AgentState) -> Dict[str, Any]:
    """생성된 답변이 문서에 근거하는지 검증한다."""
    generation = state.get("generation", "")
    documents = state.get("filtered_documents") or state.get("documents", [])

    if not documents:
        return {"steps": _add_step(state, "hallucination_check → 문서 없음, 스킵")}

    context = "\n\n---\n\n".join(documents)
    result = _llm(
        system=(
            "You are a hallucination grader. Given source documents and an answer, "
            "determine if the answer is grounded in the documents.\n"
            "Respond with ONLY a JSON object: "
            '{\"grounded\": true} or {\"grounded\": false}'
        ),
        user=f"Documents:\n{context}\n\nAnswer:\n{generation}",
    )
    try:
        grounded = json.loads(result).get("grounded", True)
    except (json.JSONDecodeError, AttributeError):
        grounded = True

    label = "통과" if grounded else "실패"
    return {"steps": _add_step(state, f"hallucination_check → {label}")}


# ── 라우팅 함수들 (조건부 엣지) ──────────────────────

def decide_route(state: AgentState) -> str:
    """route_query 결과에 따라 다음 노드를 결정."""
    last_step = state.get("steps", [])[-1] if state.get("steps") else ""
    if "direct" in last_step:
        return "direct_answer"
    return "retrieve_documents"


def decide_after_grading(state: AgentState) -> str:
    """문서 평가 후 충분한지, 재검색이 필요한지 결정."""
    filtered = state.get("filtered_documents", [])
    retry_count = state.get("retry_count", 0)

    if filtered:
        return "generate"
    if retry_count >= MAX_RETRIES:
        logger.warning("Max retries reached, generating with available docs")
        return "generate"
    return "rewrite_query"


def decide_after_hallucination(state: AgentState) -> str:
    """환각 검사 후 통과 여부에 따라 결정."""
    last_step = state.get("steps", [])[-1] if state.get("steps") else ""
    retry_count = state.get("retry_count", 0)

    if "실패" in last_step and retry_count < MAX_RETRIES:
        return "rewrite_query"
    return "end"
