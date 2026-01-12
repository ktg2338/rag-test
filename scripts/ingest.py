import glob
import os
from app.services.vectorstore import upsert_texts, get_all_documents
from app.services.chunker import chunk_text
from app.services.bm25_index import bm25_index


def load_and_ingest(pattern: str = "data/raw/**/*.*"):
    paths = [p for p in glob.glob(pattern, recursive=True) if os.path.isfile(p)]
    texts = []
    metas = []
    for p in paths:
        if p.lower().endswith((".txt", ".md")):
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            for i, ch in enumerate(chunk_text(content)):
                texts.append(ch)
                metas.append({"source": p, "chunk": i})
    if not texts:
        print("No txt/md files found in data/raw")
        return
    ids = upsert_texts(texts, metadatas=metas)
    print(f"Ingested chunks: {len(ids)}")

    # BM25 인덱스 갱신
    all_docs = get_all_documents()
    documents = all_docs.get("documents", [])
    if documents:
        bm25_index.build(documents)
        print(f"BM25 index rebuilt with {bm25_index.doc_count} documents")


if __name__ == "__main__":
    load_and_ingest()
