from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    texts: List[str]
    metadatas: Optional[List[Dict[str, Any]]] = None
    ids: Optional[List[str]] = None


class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = Field(default=4, ge=1, le=20)


class QueryResponse(BaseModel):
    answer: str
    contexts: List[str]
