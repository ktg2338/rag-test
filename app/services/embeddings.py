from typing import List
from openai import AzureOpenAI
from app.core.config import settings

_client = AzureOpenAI(
    api_key=settings.AZURE_OPENAI_API_KEY,
    azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
    api_version=settings.AZURE_OPENAI_API_VERSION,
)


def embed_texts(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    resp = _client.embeddings.create(
        model=settings.AZURE_OPENAI_EMBED_DEPLOYMENT,
        input=texts,
    )
    # resp.data 는 입력 순서대로 반환
    return [d.embedding for d in resp.data]
