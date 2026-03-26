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


class QueryResponse(BaseModel):
    answer: str
    contexts: List[str]
    conversation_id: str
    graph_entities: Optional[List[str]] = None


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
