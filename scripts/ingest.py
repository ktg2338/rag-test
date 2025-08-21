import glob
import os
from app.services.vectorstore import upsert_texts

# 매우 단순한 청크 분할 (문자 기준)
def chunk_text(text: str, size: int = 1000, overlap: int = 200):
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + size, n)
        chunks.append(text[start:end])
        if end == n:
            break
        start = end - overlap
        if start < 0:
            start = 0
    return chunks


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


if __name__ == "__main__":
    load_and_ingest()
