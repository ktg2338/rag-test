"""Claude Code에서 사용할 MCP 서버.

기존 FastAPI REST와 독립적으로 동작. 서비스 레이어(app.services.*)를 공유하며,
Claude가 LLM 역할(답변 생성, 엔티티 추출, 라우팅)을 직접 담당한다.
"""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

from app.services import vectorstore
from app.services.retriever import _retrieve_chunks
from app.services.graph_store import graph_store
from app.services.community_summarizer import (
    get_global_summaries,
    invalidate_cache,
)
from app.services.graph_extractor import extract_triples_batch

mcp = FastMCP("rag-graph")


# ── 검색 tools ────────────────────────────────────────────────


@mcp.tool()
def rag_search(query: str, top_k: int = 4) -> list[dict[str, Any]]:
    """Hybrid(벡터+BM25) 검색 + cross-encoder 재정렬로 관련 청크를 반환한다.

    일반적인 사실·내용 질문에 사용한다. 문서 본문 조각을 그대로 반환하므로
    Claude는 받은 청크를 근거로 답변하고 출처 추적이 가능하다.
    """
    docs, metas = _retrieve_chunks(query, top_k)
    return [{"text": doc, "metadata": meta} for doc, meta in zip(docs, metas)]


@mcp.tool()
def graph_find_entities(keywords: list[str], limit: int = 20) -> list[str]:
    """키워드 리스트로 Knowledge Graph에서 엔티티 이름을 매칭한다.

    대소문자 무시 exact/substring 매칭. 엔티티 정식 이름을 확인한 뒤
    graph_neighbors 호출에 넘기는 식으로 체이닝한다.
    """
    return graph_store.find_entities(keywords)[:limit]


@mcp.tool()
def graph_neighbors(
    entity: str, depth: int = 2, max_triples: int = 30
) -> list[dict[str, str]]:
    """엔티티로부터 N-hop 이웃 트리플(subject, relation, object)을 반환한다.

    "X와 관련된 것", "X의 연결 관계", "X가 어떻게 Y와 이어지는가" 같은
    관계·구조 질문에 사용한다.
    """
    triples = graph_store.get_neighbors(entity, depth=depth)[:max_triples]
    return [{"subject": s, "relation": r, "object": o} for s, r, o in triples]


@mcp.tool()
def graph_community_summary(query: str, top_k: int = 5) -> list[str]:
    """Louvain 커뮤니티 요약에서 쿼리와 유사한 상위 요약을 반환한다.

    "전체 주제 개요", "큰 흐름", "어떤 영역들이 있는가" 같은 global 질문에 사용.
    개별 청크 대신 그래프 클러스터 단위 요약을 제공한다.
    """
    return get_global_summaries(query, top_k=top_k)


# ── 주입 tools ────────────────────────────────────────────────


@mcp.tool()
def rag_ingest(
    texts: list[str], metadatas: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    """텍스트를 pgvector에 임베딩하여 저장한다. 저장된 id 리스트를 반환."""
    ids = vectorstore.upsert_texts(texts=texts, metadatas=metadatas)
    return {"inserted": len(ids), "ids": ids}


@mcp.tool()
def graph_ingest(texts: list[str] | None = None) -> dict[str, Any]:
    """텍스트에서 트리플을 추출해 Neo4j에 저장한다.

    texts 미지정 시 pgvector에 저장된 모든 문서를 대상으로 한다.
    커뮤니티 요약 캐시는 자동 무효화된다.
    """
    if texts:
        chunk_ids: list[str | None] = [None] * len(texts)
    else:
        all_docs = vectorstore.get_all_documents()
        texts = all_docs.get("documents", [])
        chunk_ids = all_docs.get("ids", [])

    if not texts:
        return {"triples": 0, "nodes": 0, "edges": 0}

    results = extract_triples_batch(texts, chunk_ids)
    total = 0
    for triples, cid in results:
        total += graph_store.add_triples(triples, chunk_id=cid)
    invalidate_cache()

    return {
        "triples": total,
        "nodes": graph_store.node_count,
        "edges": graph_store.edge_count,
    }


# ── Resources (읽기 전용 컨텍스트) ────────────────────────────


@mcp.resource("rag://stats")
def stats() -> str:
    """벡터 스토어 문서 수 및 Knowledge Graph 노드/엣지 수."""
    docs = vectorstore.get_all_documents()
    return (
        f"documents: {docs.get('count', 0)}\n"
        f"graph_nodes: {graph_store.node_count}\n"
        f"graph_edges: {graph_store.edge_count}"
    )


@mcp.resource("rag://entities")
def entities() -> str:
    """Knowledge Graph 엔티티 샘플 (최대 50개)."""
    names = graph_store.get_entity_names(limit=50)
    return "\n".join(names) if names else "(empty)"


if __name__ == "__main__":
    mcp.run(transport="stdio")
