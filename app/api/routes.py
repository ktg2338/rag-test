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
    answer, contexts, conversation_id = rag.answer_question(
        req.question, top_k=req.top_k, conversation_id=req.conversation_id
    )
    return QueryResponse(
        answer=answer, contexts=contexts, conversation_id=conversation_id
    )


@router.get("/documents", tags=["debug"])
def get_all_documents():
    """ChromaDB에 저장된 모든 문서 조회"""
    return vectorstore.get_all_documents()
