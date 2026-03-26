import json
import uuid
from typing import Any, Dict, List, Optional, Tuple

import psycopg2
from pgvector.psycopg2 import register_vector
from psycopg2.extras import execute_values

from app.core.config import settings
from app.services.embeddings import embed_texts


def _get_raw_conn():
    """vector 타입 등록 없이 연결 (확장 생성 전용)"""
    return psycopg2.connect(
        host=settings.POSTGRES_HOST,
        port=settings.POSTGRES_PORT,
        dbname=settings.POSTGRES_DB,
        user=settings.POSTGRES_USER,
        password=settings.POSTGRES_PASSWORD,
    )


def _get_conn():
    """vector 타입이 등록된 연결 (확장 생성 후 사용)"""
    conn = _get_raw_conn()
    register_vector(conn)
    return conn


def init_db() -> None:
    """pgvector 확장 및 documents 테이블 생성"""
    conn = _get_raw_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    metadata JSONB DEFAULT '{{}}'::jsonb,
                    embedding vector({settings.EMBEDDING_DIMENSION})
                )
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS documents_embedding_idx
                ON documents USING hnsw (embedding vector_cosine_ops)
            """)
        conn.commit()
    finally:
        conn.close()


def upsert_texts(
    texts: List[str],
    metadatas: Optional[List[Dict[str, Any]]] = None,
    ids: Optional[List[str]] = None,
) -> List[str]:
    if not texts:
        return []

    if ids is None:
        ids = [str(uuid.uuid4()) for _ in texts]
    if metadatas is None:
        metadatas = [{"source": "unknown"} for _ in texts]
    metadatas = [m if m else {"source": "unknown"} for m in metadatas]

    embs = embed_texts(texts)

    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            values = [
                (id_, text, json.dumps(meta, ensure_ascii=False), emb)
                for id_, text, meta, emb in zip(ids, texts, metadatas, embs)
            ]
            execute_values(
                cur,
                """
                INSERT INTO documents (id, content, metadata, embedding)
                VALUES %s
                ON CONFLICT (id) DO UPDATE SET
                    content = EXCLUDED.content,
                    metadata = EXCLUDED.metadata,
                    embedding = EXCLUDED.embedding
                """,
                values,
                template="(%s, %s, %s::jsonb, %s::vector)",
            )
        conn.commit()
    finally:
        conn.close()

    return ids


def query_similar(query_text: str, top_k: int) -> Tuple[list, list, list]:
    """returns (documents, metadatas, distances) — cosine distance"""
    q_emb = embed_texts([query_text])[0]

    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT content, metadata, embedding <=> %s::vector AS distance
                FROM documents
                ORDER BY distance
                LIMIT %s
                """,
                (q_emb, top_k),
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    docs = [row[0] for row in rows]
    metas = [row[1] if row[1] else {} for row in rows]
    dists = [float(row[2]) for row in rows]
    return docs, metas, dists


def get_all_documents() -> Dict[str, Any]:
    """저장된 모든 문서 반환"""
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id, content, metadata FROM documents")
            rows = cur.fetchall()
    finally:
        conn.close()

    return {
        "count": len(rows),
        "ids": [row[0] for row in rows],
        "documents": [row[1] for row in rows],
        "metadatas": [row[2] if row[2] else {} for row in rows],
    }
