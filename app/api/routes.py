import logging

from fastapi import APIRouter

from app.models.schemas import (
    IngestRequest,
    QueryRequest,
    QueryResponse,
    GraphIngestRequest,
    GraphIngestResponse,
    GraphStatsResponse,
)
from app.services import vectorstore
from app.services import rag

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/ingest", tags=["ingest"])
def ingest(req: IngestRequest):
    ids = vectorstore.upsert_texts(
        texts=req.texts,
        metadatas=req.metadatas,
        ids=req.ids,
    )
    return {"inserted": len(ids), "ids": ids}


@router.post("/query", response_model=QueryResponse, tags=["rag"])
def query(req: QueryRequest):
    answer, contexts, conversation_id, graph_entities = rag.answer_question(
        req.question,
        top_k=req.top_k,
        conversation_id=req.conversation_id,
        mode=req.mode or "local",
    )
    return QueryResponse(
        answer=answer,
        contexts=contexts,
        conversation_id=conversation_id,
        graph_entities=graph_entities,
    )


@router.get("/documents", tags=["debug"])
def get_all_documents():
    """저장된 모든 문서 조회 (PostgreSQL)"""
    return vectorstore.get_all_documents()


# ── GraphRAG Endpoints ──


@router.post("/graph/ingest", response_model=GraphIngestResponse, tags=["graph"])
def graph_ingest(req: GraphIngestRequest):
    """텍스트에서 트리플을 추출하여 Knowledge Graph 구축"""
    from app.services.graph_extractor import extract_triples_batch
    from app.services.graph_store import graph_store
    from app.services.community_summarizer import invalidate_cache

    # 텍스트가 없으면 PostgreSQL에서 모든 문서 로드
    if req.texts:
        texts = req.texts
        chunk_ids = req.chunk_ids
    else:
        all_docs = vectorstore.get_all_documents()
        texts = all_docs.get("documents", [])
        chunk_ids = all_docs.get("ids", [])

    if not texts:
        return GraphIngestResponse(triples_extracted=0, nodes=0, edges=0)

    # 배치 트리플 추출
    results = extract_triples_batch(texts, chunk_ids)

    total_triples = 0
    for triples, chunk_id in results:
        added = graph_store.add_triples(triples, chunk_id=chunk_id)
        total_triples += added

    # 캐시 초기화 (Neo4j는 자동 영속화)
    invalidate_cache()

    logger.info(
        "Graph ingest complete: %d triples, %d nodes, %d edges",
        total_triples,
        graph_store.node_count,
        graph_store.edge_count,
    )

    return GraphIngestResponse(
        triples_extracted=total_triples,
        nodes=graph_store.node_count,
        edges=graph_store.edge_count,
    )


@router.get("/graph/stats", response_model=GraphStatsResponse, tags=["graph"])
def graph_stats():
    """Knowledge Graph 통계"""
    from app.services.graph_store import graph_store
    from app.services.community_summarizer import detect_communities

    communities = detect_communities() if graph_store.node_count >= 2 else []
    return GraphStatsResponse(
        nodes=graph_store.node_count,
        edges=graph_store.edge_count,
        communities=len(communities),
    )


@router.get("/graph/entities", tags=["graph"])
def graph_entities(q: str = "", limit: int = 50):
    """그래프 엔티티 검색/조회"""
    from app.services.graph_store import graph_store

    if q:
        matched = graph_store.find_entities([q])
        # 매칭된 엔티티의 이웃 트리플도 반환
        triples = []
        for entity in matched[:limit]:
            for t in graph_store.get_neighbors(entity, depth=1):
                triples.append({"subject": t[0], "relation": t[1], "object": t[2]})
        return {"matched_entities": matched[:limit], "triples": triples}

    # 전체 노드 목록
    nodes = graph_store.get_entity_names(limit=limit)
    return {"entities": nodes, "total": graph_store.node_count}
