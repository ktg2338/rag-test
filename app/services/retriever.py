from typing import List, Dict, Any, Tuple
from app.core.config import settings
from app.services.vectorstore import query_similar


def retrieve(query: str, top_k: int | None = None) -> Tuple[List[str], List[Dict[str, Any]]]:
    k = top_k or settings.MAX_CONTEXT_CHUNKS
    docs, metas, _ = query_similar(query, k)
    return docs, metas
