import uuid
from typing import List, Optional, Dict, Any, Tuple

import chromadb
from app.core.config import settings
from app.services.embeddings import embed_texts

# 퍼시스턴트 클라이언트
_client = chromadb.PersistentClient(path=settings.CHROMA_PATH)
# cosine 유사도(HNSW 기본)
_collection = _client.get_or_create_collection(
    name="docs",
    metadata={"hnsw:space": "cosine"},
)


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
        metadatas = [{} for _ in texts]

    embs = embed_texts(texts)
    _collection.upsert(
        ids=ids,
        documents=texts,
        metadatas=metadatas,
        embeddings=embs,
    )
    return ids


def query_similar(query_text: str, top_k: int) -> Tuple[list, list, list]:
    """returns (documents, metadatas, distances)"""
    q_emb = embed_texts([query_text])[0]
    res = _collection.query(
        query_embeddings=[q_emb],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    return docs, metas, dists
