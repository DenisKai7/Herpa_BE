"""
Konfigurasi Aplikasi Enterprise - Pharmaceutical AI Backend.
Menggunakan pydantic-settings untuk memuat environment variables secara type-safe.
"""

from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Pengaturan global aplikasi yang dimuat dari .env file."""

    # ─── IDENTITAS APLIKASI ───
    APP_NAME: str = "Enterprise GraphRAG Agentic AI"
    APP_VERSION: str = "2.0.0"
    DEBUG: bool = False

    # ─── SUPABASE (PostgreSQL + pgvector + Auth) ───
    SUPABASE_URL: str
    SUPABASE_SERVICE_KEY: str  # Service role key untuk operasi backend

    # ─── NEO4J (Graph Database) ───
    NEO4J_URI: str = "bolt://medical_neo4j:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "changeme"

    # ─── REDIS (Rate Limiting) ───
    REDIS_URL: str = "redis://medical_redis:6379"

    # ─── MINIO (Object Storage) ───
    MINIO_ENDPOINT: str = "medical_minio:9000"
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str = "minioadmin"
    MINIO_BUCKET: str = "chat-attachments"
    MINIO_SECURE: bool = False

    # ─── HUGGINGFACE ───
    HF_API_TOKEN: str  # HuggingFace API token

    # ─── LLM (via HuggingFace Inference API) ───
    LLM_BASE_URL: str = "https://router.huggingface.co/v1"
    LLM_MODEL: str = "meta-llama/Llama-3.2-3B-Instruct"

    # ─── EMBEDDING MODEL (via HuggingFace Inference API) ───
    EMBEDDING_MODEL_NAME: str = "intfloat/multilingual-e5-base"

    # ─── CORS ───
    CORS_ORIGINS: str = "*"  # Comma-separated, e.g. "http://localhost:3000,https://myapp.com"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """Singleton cached settings instance."""
    return Settings()


# Convenience: importable instance
settings = get_settings()
