from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    AZURE_OPENAI_API_KEY: str
    AZURE_OPENAI_ENDPOINT: str
    AZURE_OPENAI_API_VERSION: str = "2024-02-15-preview"
    AZURE_OPENAI_DEPLOYMENT: str
    AZURE_OPENAI_EMBED_DEPLOYMENT: str
    CHROMA_PATH: str = "data/chroma"
    MAX_CONTEXT_CHUNKS: int = 4

    # Hybrid Search 설정
    HYBRID_SEARCH_ENABLED: bool = True
    BM25_WEIGHT: float = 0.3  # BM25 점수 가중치 (1 - BM25_WEIGHT = vector 가중치)
    HYBRID_CANDIDATE_MULTIPLIER: int = 3  # top_k * multiplier = 후보 문서 수

    # Reranking 설정
    RERANKER_ENABLED: bool = True

    # Semantic Cache 설정
    SEMANTIC_CACHE_ENABLED: bool = True
    SEMANTIC_CACHE_THRESHOLD: float = 0.95  # 코사인 유사도 (1 - distance)

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
