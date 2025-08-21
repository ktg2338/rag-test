from typing import List
from openai import OpenAI
from app.core.config import settings

_client = OpenAI(api_key=settings.OPENAI_API_KEY)


def embed_texts(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    resp = _client.embeddings.create(
        model=settings.OPENAI_EMBED_MODEL,
        input=texts
    )
    # resp.data 는 입력 순서대로 반환
    return [d.embedding for d in resp.data]
