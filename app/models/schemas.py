from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    texts: List[str]
    metadatas: Optional[List[Dict[str, Any]]] = None
    ids: Optional[List[str]] = None


class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = Field(default=4, ge=1, le=20)
    conversation_id: Optional[str] = None
    mode: Optional[str] = Field(
        default="local",
        pattern=r"^(local|global|hybrid)$",
        description="local: chunk+graph, global: community summaries, hybrid: both",
    )
    use_cache: bool = True
    cache_threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="코사인 유사도 임계값. 미지정 시 SEMANTIC_CACHE_THRESHOLD 사용.",
    )


class QueryResponse(BaseModel):
    answer: str
    contexts: List[str]
    conversation_id: str
    graph_entities: Optional[List[str]] = None
    cache_hit: bool = False
    cache_similarity: Optional[float] = None


# ── GraphRAG ──


class GraphIngestRequest(BaseModel):
    texts: Optional[List[str]] = None
    chunk_ids: Optional[List[str]] = None


class GraphIngestResponse(BaseModel):
    triples_extracted: int
    nodes: int
    edges: int


class GraphStatsResponse(BaseModel):
    nodes: int
    edges: int
    communities: int


# ── Semantic Cache ──


class CacheStatsResponse(BaseModel):
    entries: int
    total_hits: int
    oldest_created_at: Optional[str] = None
    last_accessed_at: Optional[str] = None
    threshold: float
    enabled: bool


class CacheEntry(BaseModel):
    id: str
    question: str
    answer: str
    contexts: List[str]
    graph_entities: List[str]
    hit_count: int
    created_at: Optional[str] = None
    last_accessed_at: Optional[str] = None


class CacheClearResponse(BaseModel):
    deleted: int
