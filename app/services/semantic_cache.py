"""Semantic cache backed by pgvector.

질문 임베딩의 코사인 유사도가 임계값 이상이면 저장된 답변을 그대로 반환한다.
"""

import json
import logging
import uuid
from typing import Any, Dict, List, Optional, TypedDict

import psycopg2
from pgvector.psycopg2 import register_vector
from psycopg2.extras import execute_values

from app.core.config import settings
from app.services.embeddings import embed_texts

logger = logging.getLogger(__name__)


class CacheHit(TypedDict):
    id: str
    question: str
    answer: str
    contexts: List[str]
    graph_entities: List[str]
    similarity: float
    hit_count: int


def _get_conn():
    conn = psycopg2.connect(
        host=settings.POSTGRES_HOST,
        port=settings.POSTGRES_PORT,
        dbname=settings.POSTGRES_DB,
        user=settings.POSTGRES_USER,
        password=settings.POSTGRES_PASSWORD,
    )
    register_vector(conn)
    return conn


def init_cache_table() -> None:
    """semantic_cache 테이블 + HNSW 인덱스 생성."""
    conn = psycopg2.connect(
        host=settings.POSTGRES_HOST,
        port=settings.POSTGRES_PORT,
        dbname=settings.POSTGRES_DB,
        user=settings.POSTGRES_USER,
        password=settings.POSTGRES_PASSWORD,
    )
    try:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS semantic_cache (
                    id TEXT PRIMARY KEY,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    contexts JSONB DEFAULT '[]'::jsonb,
                    graph_entities JSONB DEFAULT '[]'::jsonb,
                    embedding vector({settings.EMBEDDING_DIMENSION}),
                    hit_count INTEGER NOT NULL DEFAULT 0,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    last_accessed_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS semantic_cache_embedding_idx
                ON semantic_cache USING hnsw (embedding vector_cosine_ops)
            """)
        conn.commit()
    finally:
        conn.close()


def lookup(question: str, threshold: Optional[float] = None) -> Optional[CacheHit]:
    """가장 유사한 캐시 항목을 조회. similarity >= threshold면 hit.

    threshold가 None이면 settings.SEMANTIC_CACHE_THRESHOLD 사용.
    hit 시 hit_count, last_accessed_at을 갱신한다.
    """
    if threshold is None:
        threshold = settings.SEMANTIC_CACHE_THRESHOLD

    q_emb = embed_texts([question])[0]
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, question, answer, contexts, graph_entities,
                       embedding <=> %s::vector AS distance, hit_count
                FROM semantic_cache
                ORDER BY distance
                LIMIT 1
                """,
                (q_emb,),
            )
            row = cur.fetchone()
            if not row:
                return None

            id_, cached_q, answer, contexts, graph_entities, distance, hit_count = row
            similarity = 1.0 - float(distance)
            if similarity < threshold:
                return None

            cur.execute(
                """
                UPDATE semantic_cache
                SET hit_count = hit_count + 1,
                    last_accessed_at = NOW()
                WHERE id = %s
                """,
                (id_,),
            )
            conn.commit()
    finally:
        conn.close()

    return CacheHit(
        id=id_,
        question=cached_q,
        answer=answer,
        contexts=contexts or [],
        graph_entities=graph_entities or [],
        similarity=similarity,
        hit_count=hit_count + 1,
    )


def store(
    question: str,
    answer: str,
    contexts: List[str],
    graph_entities: Optional[List[str]] = None,
) -> str:
    """질문/답변을 캐시에 저장."""
    cache_id = str(uuid.uuid4())
    emb = embed_texts([question])[0]
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            execute_values(
                cur,
                """
                INSERT INTO semantic_cache
                  (id, question, answer, contexts, graph_entities, embedding)
                VALUES %s
                """,
                [(
                    cache_id,
                    question,
                    answer,
                    json.dumps(contexts, ensure_ascii=False),
                    json.dumps(graph_entities or [], ensure_ascii=False),
                    emb,
                )],
                template="(%s, %s, %s, %s::jsonb, %s::jsonb, %s::vector)",
            )
        conn.commit()
    finally:
        conn.close()
    return cache_id


def clear() -> int:
    """모든 캐시 항목 삭제. 삭제된 행 수 반환."""
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM semantic_cache")
            deleted = cur.rowcount
        conn.commit()
    finally:
        conn.close()
    return deleted


def delete(cache_id: str) -> bool:
    """단건 삭제."""
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM semantic_cache WHERE id = %s", (cache_id,))
            deleted = cur.rowcount
        conn.commit()
    finally:
        conn.close()
    return deleted > 0


def stats() -> Dict[str, Any]:
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT COUNT(*),
                       COALESCE(SUM(hit_count), 0),
                       MIN(created_at),
                       MAX(last_accessed_at)
                FROM semantic_cache
                """
            )
            entries, total_hits, oldest, newest_access = cur.fetchone()
    finally:
        conn.close()
    return {
        "entries": int(entries),
        "total_hits": int(total_hits),
        "oldest_created_at": oldest.isoformat() if oldest else None,
        "last_accessed_at": newest_access.isoformat() if newest_access else None,
        "threshold": settings.SEMANTIC_CACHE_THRESHOLD,
        "enabled": settings.SEMANTIC_CACHE_ENABLED,
    }


def list_entries(limit: int = 50) -> List[Dict[str, Any]]:
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, question, answer, contexts, graph_entities,
                       hit_count, created_at, last_accessed_at
                FROM semantic_cache
                ORDER BY last_accessed_at DESC
                LIMIT %s
                """,
                (limit,),
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    return [
        {
            "id": r[0],
            "question": r[1],
            "answer": r[2],
            "contexts": r[3] or [],
            "graph_entities": r[4] or [],
            "hit_count": r[5],
            "created_at": r[6].isoformat() if r[6] else None,
            "last_accessed_at": r[7].isoformat() if r[7] else None,
        }
        for r in rows
    ]
