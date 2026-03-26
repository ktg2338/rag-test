import logging
from typing import List, Dict, Any, Tuple

from app.core.config import settings
from app.services.vectorstore import query_similar, get_all_documents
from app.services.bm25_index import bm25_index
from app.services.reranker import rerank

logger = logging.getLogger(__name__)


def _normalize_scores(scores: List[float]) -> List[float]:
    """점수를 0-1 범위로 정규화"""
    if not scores:
        return []
    min_s, max_s = min(scores), max(scores)
    if max_s == min_s:
        return [1.0] * len(scores)
    return [(s - min_s) / (max_s - min_s) for s in scores]


def _ensure_bm25_index() -> None:
    """BM25 인덱스가 비어있으면 ChromaDB에서 문서를 로드하여 구축"""
    if bm25_index.doc_count == 0:
        all_docs = get_all_documents()
        documents = all_docs.get("documents", [])
        if documents:
            bm25_index.build(documents)


def _retrieve_chunks(query: str, k: int) -> Tuple[List[str], List[Dict[str, Any]]]:
    """기존 Hybrid Search + Reranking 파이프라인"""
    # Hybrid Search가 비활성화면 기존 방식 사용
    if not settings.HYBRID_SEARCH_ENABLED:
        docs, metas, _ = query_similar(query, k)
        if settings.RERANKER_ENABLED and docs:
            reranked = rerank(query, docs, top_k=k)
            docs = [doc for doc, _ in reranked]
            doc_to_meta = dict(zip(docs, metas))
            metas = [doc_to_meta.get(doc, {}) for doc in docs]
        return docs, metas

    candidate_k = k * settings.HYBRID_CANDIDATE_MULTIPLIER

    # 1. Vector Search
    vec_docs, vec_metas, vec_distances = query_similar(query, candidate_k)
    vec_scores = [1 - d for d in vec_distances]

    # 2. BM25 Search
    _ensure_bm25_index()
    bm25_results = bm25_index.search(query, top_k=candidate_k)

    # 3. Hybrid Fusion
    doc_scores: Dict[str, float] = {}
    doc_metas: Dict[str, Dict[str, Any]] = {}

    norm_vec_scores = _normalize_scores(vec_scores)
    vec_weight = 1 - settings.BM25_WEIGHT
    for doc, meta, score in zip(vec_docs, vec_metas, norm_vec_scores):
        doc_scores[doc] = score * vec_weight
        doc_metas[doc] = meta

    bm25_docs = [bm25_index.get_document(idx) for idx, _ in bm25_results]
    bm25_raw_scores = [score for _, score in bm25_results]
    norm_bm25_scores = _normalize_scores(bm25_raw_scores)

    for doc, score in zip(bm25_docs, norm_bm25_scores):
        if doc in doc_scores:
            doc_scores[doc] += score * settings.BM25_WEIGHT
        else:
            doc_scores[doc] = score * settings.BM25_WEIGHT
            doc_metas[doc] = {}

    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    candidate_docs = [doc for doc, _ in sorted_docs[:candidate_k]]

    # 4. Reranking
    if settings.RERANKER_ENABLED and candidate_docs:
        reranked = rerank(query, candidate_docs, top_k=k)
        final_docs = [doc for doc, _ in reranked]
    else:
        final_docs = candidate_docs[:k]

    final_metas = [doc_metas.get(doc, {}) for doc in final_docs]
    return final_docs, final_metas


def retrieve(
    query: str, top_k: int | None = None, mode: str = "local"
) -> Tuple[List[str], List[Dict[str, Any]], List[str]]:
    """
    문서 검색. mode에 따라 Graph 컨텍스트를 결합.

    mode:
      - "local": Hybrid Search + Graph neighbor context
      - "global": Community summary만 반환
      - "hybrid": Hybrid Search + Graph neighbor + Community summary

    Returns: (documents, metadatas, graph_entities)
    """
    k = top_k or settings.MAX_CONTEXT_CHUNKS
    graph_entities: List[str] = []

    # Global mode: 커뮤니티 요약만 반환
    if mode == "global" and settings.GRAPH_ENABLED:
        from app.services.graph_retriever import retrieve_global_context

        global_contexts = retrieve_global_context(query, top_k=k)
        global_metas = [{"source": "graph_community"} for _ in global_contexts]
        return global_contexts, global_metas, graph_entities

    # Local / Hybrid: 기존 chunk 검색
    final_docs, final_metas = _retrieve_chunks(query, k)

    # Graph context 추가
    if settings.GRAPH_ENABLED:
        from app.services.graph_retriever import (
            retrieve_graph_context,
            retrieve_global_context,
        )

        # Local graph context (entity neighbor traversal)
        graph_contexts, matched = retrieve_graph_context(
            query, max_triples=settings.GRAPH_MAX_CONTEXT_TRIPLES
        )
        graph_entities = matched

        if graph_contexts:
            graph_metas = [{"source": "graph"} for _ in graph_contexts]
            # Graph 컨텍스트를 앞에 배치 (구조화된 지식 우선)
            final_docs = graph_contexts + final_docs
            final_metas = graph_metas + final_metas

        # Hybrid mode: 커뮤니티 요약도 추가
        if mode == "hybrid":
            global_contexts = retrieve_global_context(query, top_k=3)
            if global_contexts:
                global_metas = [{"source": "graph_community"} for _ in global_contexts]
                final_docs = global_contexts + final_docs
                final_metas = global_metas + final_metas

    return final_docs, final_metas, graph_entities
