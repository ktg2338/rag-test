from fastapi import FastAPI
from app.api.routes import router

app = FastAPI(title="FastAPI RAG (OpenAI + Chroma)")
app.include_router(router)

@app.get("/", tags=["health"])
def health():
    return {"status": "ok"}
