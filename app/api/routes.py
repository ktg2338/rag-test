from fastapi import APIRouter
from app.models.schemas import IngestRequest, QueryRequest, QueryResponse
from app.services import vectorstore
from app.services import rag

router = APIRouter()

@router.post("/ingest", tags=["ingest"])
def ingest(req: IngestRequest):
    ids = vectorstore.upsert_texts(
        texts=req.texts,
        metadatas=req.metadatas,
        ids=req.ids,
    )
    return {"inserted": len(ids), "ids": ids}


@router.post("/query", response_model=QueryResponse, tags=["rag"])
def query(req: QueryRequest):
    answer, contexts = rag.answer_question(req.question, top_k=req.top_k)
    return QueryResponse(answer=answer, contexts=contexts)
