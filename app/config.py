
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from .env file or environment variables."""

    # Database
    DATABASE_URL: str = "sqlite:///./doc_ingestion.db"

    # Qdrant vector store
    QDRANT_PATH: str = "./qdrant_data"
    COLLECTION_NAME: str = "document_chunks"

    # Embedding model
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384

    # Chunking defaults
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50

    # File upload
    UPLOAD_DIR: str = "./uploads"
    MAX_FILE_SIZE_MB: int = 50
    ALLOWED_EXTENSIONS: set = {".pdf", ".txt"}

    # OpenAI
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4o-mini"

    # Redis
    REDIS_URL: str = "redis://localhost:6379"
    CHAT_HISTORY_TTL: int = 3600  # seconds (1 hour)

    # RAG pipeline
    RAG_TOP_K: int = 5
    MAX_HISTORY_MESSAGES: int = 20

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


@lru_cache
def get_settings() -> Settings:
    """Return cached settings instance (singleton)."""
    return Settings()

