from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    AZURE_OPENAI_API_KEY: str
    AZURE_OPENAI_ENDPOINT: str
    AZURE_OPENAI_API_VERSION: str = "2024-02-15-preview"
    AZURE_OPENAI_DEPLOYMENT: str
    AZURE_OPENAI_EMBED_DEPLOYMENT: str
    EMBEDDING_DIMENSION: int = 1536

    # PostgreSQL + pgvector
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "ragdb"
    POSTGRES_USER: str = "rag"
    POSTGRES_PASSWORD: str = "rag"

    # Neo4j
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "neo4j"

    MAX_CONTEXT_CHUNKS: int = 4

    # Hybrid Search 설정
    HYBRID_SEARCH_ENABLED: bool = True
    BM25_WEIGHT: float = 0.3  # BM25 점수 가중치 (1 - BM25_WEIGHT = vector 가중치)
    HYBRID_CANDIDATE_MULTIPLIER: int = 3  # top_k * multiplier = 후보 문서 수

    # Reranking 설정
    RERANKER_ENABLED: bool = True

    # GraphRAG 설정
    GRAPH_ENABLED: bool = True
    GRAPH_MAX_TRIPLES_PER_CHUNK: int = 20
    GRAPH_NEIGHBOR_DEPTH: int = 2
    GRAPH_MAX_CONTEXT_TRIPLES: int = 30
    GRAPH_WEIGHT: float = 0.3
    GRAPH_COMMUNITY_MIN_SIZE: int = 3
    GRAPH_COMMUNITY_RESOLUTION: float = 1.0

    # Semantic Cache 설정
    SEMANTIC_CACHE_ENABLED: bool = True
    SEMANTIC_CACHE_THRESHOLD: float = 0.95  # 코사인 유사도 (1 - distance)

    class Config:
        env_file = ".env"


settings = Settings()
