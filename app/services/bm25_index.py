import re
from typing import List, Tuple, Optional
from rank_bm25 import BM25Okapi


class BM25Index:
    """BM25 키워드 검색 인덱스"""

    def __init__(self):
        self._documents: List[str] = []
        self._tokenized_docs: List[List[str]] = []
        self._bm25: Optional[BM25Okapi] = None

    def _tokenize(self, text: str) -> List[str]:
        """간단한 토크나이저: 한글/영문/숫자 단위로 분리"""
        text = text.lower()
        tokens = re.findall(r"[가-힣]+|[a-z]+|[0-9]+", text)
        return tokens

    def build(self, documents: List[str]) -> None:
        """문서 리스트로 BM25 인덱스 구축"""
        self._documents = documents
        self._tokenized_docs = [self._tokenize(doc) for doc in documents]
        if self._tokenized_docs:
            self._bm25 = BM25Okapi(self._tokenized_docs)

    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        BM25 검색 수행.
        Returns: [(doc_index, score), ...] 점수 내림차순
        """
        if self._bm25 is None or not self._documents:
            return []

        tokenized_query = self._tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)

        # (index, score) 쌍으로 정렬
        indexed_scores = [(i, float(scores[i])) for i in range(len(scores))]
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        return indexed_scores[:top_k]

    def get_document(self, index: int) -> str:
        """인덱스로 문서 조회"""
        if 0 <= index < len(self._documents):
            return self._documents[index]
        return ""

    @property
    def doc_count(self) -> int:
        return len(self._documents)


# 싱글톤 인스턴스
bm25_index = BM25Index()
