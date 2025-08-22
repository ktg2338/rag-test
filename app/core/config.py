from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    AZURE_OPENAI_API_KEY: str
    AZURE_OPENAI_ENDPOINT: str
    AZURE_OPENAI_API_VERSION: str = "2024-02-15-preview"
    AZURE_OPENAI_DEPLOYMENT: str
    AZURE_OPENAI_EMBED_DEPLOYMENT: str
    CHROMA_PATH: str = "data/chroma"
    MAX_CONTEXT_CHUNKS: int = 4

    class Config:
        env_file = ".env"


settings = Settings()
