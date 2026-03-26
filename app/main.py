import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    from app.services.vectorstore import init_db
    from app.services.graph_store import graph_store

    init_db()
    graph_store.verify_connection()
    yield
    graph_store.close()


app = FastAPI(title="FastAPI RAG (pgvector + Neo4j)", lifespan=lifespan)
app.include_router(router)


@app.get("/", tags=["health"])
def health():
    return {"status": "ok"}
