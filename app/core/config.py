from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    OPENAI_API_KEY: str
    OPENAI_MODEL: str = "gpt-4o-mini"
    OPENAI_EMBED_MODEL: str = "text-embedding-3-small"
    CHROMA_PATH: str = "data/chroma"
    MAX_CONTEXT_CHUNKS: int = 4

    class Config:
        env_file = ".env"


settings = Settings()
