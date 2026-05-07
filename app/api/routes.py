from fastapi import APIRouter, HTTPException

from app.models.schemas import (
    CacheClearResponse,
    CacheEntry,
    CacheStatsResponse,
    IngestRequest,
    QueryRequest,
    QueryResponse,
)
from app.services import rag, semantic_cache, vectorstore

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
    answer, contexts, conversation_id, cache_hit, cache_similarity = (
        rag.answer_question(
            req.question,
            top_k=req.top_k,
            conversation_id=req.conversation_id,
            use_cache=req.use_cache,
            cache_threshold=req.cache_threshold,
        )
    )
    return QueryResponse(
        answer=answer,
        contexts=contexts,
        conversation_id=conversation_id,
        cache_hit=cache_hit,
        cache_similarity=cache_similarity,
    )


@router.get("/documents", tags=["debug"])
def get_all_documents():
    """ChromaDB에 저장된 모든 문서 조회"""
    return vectorstore.get_all_documents()


# ── Semantic Cache Endpoints ──


@router.get("/cache/stats", response_model=CacheStatsResponse, tags=["cache"])
def cache_stats():
    return CacheStatsResponse(**semantic_cache.stats())


@router.get("/cache/entries", response_model=list[CacheEntry], tags=["cache"])
def cache_entries(limit: int = 50):
    return [CacheEntry(**e) for e in semantic_cache.list_entries(limit=limit)]


@router.post("/cache/clear", response_model=CacheClearResponse, tags=["cache"])
def cache_clear():
    return CacheClearResponse(deleted=semantic_cache.clear())


@router.delete("/cache/{cache_id}", tags=["cache"])
def cache_delete(cache_id: str):
    if not semantic_cache.delete(cache_id):
        raise HTTPException(status_code=404, detail="cache entry not found")
    return {"deleted": cache_id}
