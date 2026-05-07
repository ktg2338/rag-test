"""Semantic cache backed by ChromaDB.

질문 임베딩의 코사인 유사도가 임계값 이상이면 저장된 답변을 그대로 반환한다.
별도 컬렉션("semantic_cache")을 사용하며, RAG 문서 컬렉션과 분리된다.
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, TypedDict

import chromadb

from app.core.config import settings
from app.services.embeddings import embed_texts

logger = logging.getLogger(__name__)

_client = chromadb.PersistentClient(path=settings.CHROMA_PATH)
_collection = _client.get_or_create_collection(
    name="semantic_cache",
    metadata={"hnsw:space": "cosine"},
)


class CacheHit(TypedDict):
    id: str
    question: str
    answer: str
    contexts: List[str]
    similarity: float
    hit_count: int


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def lookup(question: str, threshold: Optional[float] = None) -> Optional[CacheHit]:
    """가장 유사한 캐시 항목을 조회. similarity >= threshold면 hit.

    threshold가 None이면 settings.SEMANTIC_CACHE_THRESHOLD 사용.
    hit 시 hit_count, last_accessed_at을 갱신한다.
    """
    if threshold is None:
        threshold = settings.SEMANTIC_CACHE_THRESHOLD

    if _collection.count() == 0:
        return None

    q_emb = embed_texts([question])[0]
    res = _collection.query(
        query_embeddings=[q_emb],
        n_results=1,
        include=["documents", "metadatas", "distances"],
    )

    ids = res.get("ids", [[]])[0]
    if not ids:
        return None

    cache_id = ids[0]
    cached_q = res.get("documents", [[]])[0][0]
    meta = res.get("metadatas", [[]])[0][0] or {}
    distance = float(res.get("distances", [[]])[0][0])
    similarity = 1.0 - distance
    if similarity < threshold:
        return None

    answer = meta.get("answer", "")
    contexts_json = meta.get("contexts", "[]")
    try:
        contexts = json.loads(contexts_json) if isinstance(contexts_json, str) else []
    except json.JSONDecodeError:
        contexts = []
    hit_count = int(meta.get("hit_count", 0)) + 1

    new_meta = {
        **meta,
        "hit_count": hit_count,
        "last_accessed_at": _now_iso(),
    }
    _collection.update(ids=[cache_id], metadatas=[new_meta])

    return CacheHit(
        id=cache_id,
        question=cached_q,
        answer=answer,
        contexts=contexts,
        similarity=similarity,
        hit_count=hit_count,
    )


def store(question: str, answer: str, contexts: List[str]) -> str:
    """질문/답변을 캐시에 저장."""
    cache_id = str(uuid.uuid4())
    emb = embed_texts([question])[0]
    now = _now_iso()
    _collection.add(
        ids=[cache_id],
        documents=[question],
        embeddings=[emb],
        metadatas=[{
            "answer": answer,
            "contexts": json.dumps(contexts, ensure_ascii=False),
            "hit_count": 0,
            "created_at": now,
            "last_accessed_at": now,
        }],
    )
    return cache_id


def clear() -> int:
    """모든 캐시 항목 삭제. 삭제된 행 수 반환."""
    ids = _collection.get().get("ids", [])
    if ids:
        _collection.delete(ids=ids)
    return len(ids)


def delete(cache_id: str) -> bool:
    """단건 삭제."""
    existing = _collection.get(ids=[cache_id]).get("ids", [])
    if not existing:
        return False
    _collection.delete(ids=[cache_id])
    return True


def stats() -> Dict[str, Any]:
    res = _collection.get(include=["metadatas"])
    metas = res.get("metadatas", []) or []
    total_hits = sum(int((m or {}).get("hit_count", 0)) for m in metas)
    created = [(m or {}).get("created_at") for m in metas if m and m.get("created_at")]
    accessed = [
        (m or {}).get("last_accessed_at")
        for m in metas
        if m and m.get("last_accessed_at")
    ]
    return {
        "entries": len(metas),
        "total_hits": total_hits,
        "oldest_created_at": min(created) if created else None,
        "last_accessed_at": max(accessed) if accessed else None,
        "threshold": settings.SEMANTIC_CACHE_THRESHOLD,
        "enabled": settings.SEMANTIC_CACHE_ENABLED,
    }


def list_entries(limit: int = 50) -> List[Dict[str, Any]]:
    res = _collection.get(include=["documents", "metadatas"])
    ids = res.get("ids", [])
    docs = res.get("documents", [])
    metas = res.get("metadatas", [])

    rows = []
    for cid, q, m in zip(ids, docs, metas):
        m = m or {}
        contexts_json = m.get("contexts", "[]")
        try:
            contexts = (
                json.loads(contexts_json) if isinstance(contexts_json, str) else []
            )
        except json.JSONDecodeError:
            contexts = []
        rows.append({
            "id": cid,
            "question": q,
            "answer": m.get("answer", ""),
            "contexts": contexts,
            "hit_count": int(m.get("hit_count", 0)),
            "created_at": m.get("created_at"),
            "last_accessed_at": m.get("last_accessed_at"),
        })

    rows.sort(key=lambda r: r["last_accessed_at"] or "", reverse=True)
    return rows[:limit]
