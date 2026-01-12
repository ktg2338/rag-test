from typing import List, Tuple
from sentence_transformers import CrossEncoder

# Cross-encoder 모델 (lazy loading)
_reranker: CrossEncoder | None = None


def _get_reranker() -> CrossEncoder:
    """Cross-encoder 모델 lazy loading"""
    global _reranker
    if _reranker is None:
        # ms-marco-MiniLM: 빠르고 효과적인 reranking 모델
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker


def rerank(
    query: str,
    documents: List[str],
    top_k: int | None = None,
) -> List[Tuple[str, float]]:
    """
    Cross-encoder를 사용하여 문서 재정렬.

    Args:
        query: 검색 쿼리
        documents: 재정렬할 문서 리스트
        top_k: 반환할 상위 문서 수 (None이면 전체)

    Returns:
        [(document, score), ...] 점수 내림차순
    """
    if not documents:
        return []

    model = _get_reranker()

    # Cross-encoder 입력: (query, document) 쌍
    pairs = [(query, doc) for doc in documents]
    scores = model.predict(pairs)

    # (document, score) 쌍으로 정렬
    doc_scores = list(zip(documents, scores))
    doc_scores.sort(key=lambda x: x[1], reverse=True)

    if top_k is not None:
        doc_scores = doc_scores[:top_k]

    return [(doc, float(score)) for doc, score in doc_scores]
