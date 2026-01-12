from typing import List, Dict, Any, Tuple
from app.core.config import settings
from app.services.vectorstore import query_similar, get_all_documents
from app.services.bm25_index import bm25_index
from app.services.reranker import rerank


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


def retrieve(query: str, top_k: int | None = None) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Hybrid Search + Reranking을 적용한 문서 검색.

    1. Vector Search (ChromaDB): 시맨틱 유사도 기반
    2. BM25 Search: 키워드 매칭 기반
    3. Hybrid Fusion: 두 결과를 가중치로 결합
    4. Reranking: Cross-encoder로 최종 정렬
    """
    k = top_k or settings.MAX_CONTEXT_CHUNKS

    # Hybrid Search가 비활성화면 기존 방식 사용
    if not settings.HYBRID_SEARCH_ENABLED:
        docs, metas, _ = query_similar(query, k)
        if settings.RERANKER_ENABLED and docs:
            reranked = rerank(query, docs, top_k=k)
            docs = [doc for doc, _ in reranked]
            # 메타데이터 순서도 맞춰줌
            doc_to_meta = dict(zip(docs, metas))
            metas = [doc_to_meta.get(doc, {}) for doc in docs]
        return docs, metas

    # 후보 문서 수 (reranking을 위해 더 많이 가져옴)
    candidate_k = k * settings.HYBRID_CANDIDATE_MULTIPLIER

    # 1. Vector Search
    vec_docs, vec_metas, vec_distances = query_similar(query, candidate_k)

    # cosine distance를 similarity로 변환 (ChromaDB는 distance 반환)
    vec_scores = [1 - d for d in vec_distances]  # cosine distance -> similarity

    # 2. BM25 Search
    _ensure_bm25_index()
    bm25_results = bm25_index.search(query, top_k=candidate_k)

    # 3. Hybrid Fusion (RRF 또는 가중 합산)
    # 문서별 점수 맵 구축
    doc_scores: Dict[str, float] = {}
    doc_metas: Dict[str, Dict[str, Any]] = {}

    # Vector 점수 정규화 및 추가
    norm_vec_scores = _normalize_scores(vec_scores)
    vec_weight = 1 - settings.BM25_WEIGHT
    for doc, meta, score in zip(vec_docs, vec_metas, norm_vec_scores):
        doc_scores[doc] = score * vec_weight
        doc_metas[doc] = meta

    # BM25 점수 정규화 및 추가
    bm25_docs = [bm25_index.get_document(idx) for idx, _ in bm25_results]
    bm25_raw_scores = [score for _, score in bm25_results]
    norm_bm25_scores = _normalize_scores(bm25_raw_scores)

    for doc, score in zip(bm25_docs, norm_bm25_scores):
        if doc in doc_scores:
            doc_scores[doc] += score * settings.BM25_WEIGHT
        else:
            doc_scores[doc] = score * settings.BM25_WEIGHT
            # BM25에만 있는 문서는 메타데이터가 없음
            doc_metas[doc] = {}

    # 점수순 정렬
    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    candidate_docs = [doc for doc, _ in sorted_docs[:candidate_k]]

    # 4. Reranking
    if settings.RERANKER_ENABLED and candidate_docs:
        reranked = rerank(query, candidate_docs, top_k=k)
        final_docs = [doc for doc, _ in reranked]
    else:
        final_docs = candidate_docs[:k]

    # 메타데이터 매핑
    final_metas = [doc_metas.get(doc, {}) for doc in final_docs]

    return final_docs, final_metas
