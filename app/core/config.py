"""
Konfigurasi Aplikasi Enterprise - Pharmaceutical AI Backend.
"""

from functools import lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Pengaturan global aplikasi."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # Aplikasi
    APP_NAME: str = "Enterprise GraphRAG Agentic AI"
    APP_VERSION: str = "2.0.0"
    DEBUG: bool = False

    # Supabase
    SUPABASE_URL: str
    SUPABASE_SERVICE_KEY: str

    # Neo4j
    NEO4J_URI: str = "bolt://medical_neo4j:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "changeme"

    # Redis
    REDIS_URL: str = "redis://medical_redis:6379"

    # MinIO
    MINIO_ENDPOINT: str = "medical-minio:9000"
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str = "minioadmin"
    MINIO_BUCKET: str = "chat-attachments"
    MINIO_SECURE: bool = False
    MINIO_PUBLIC_ENDPOINT: str = "localhost:9000"

    # Hugging Face
    HF_API_TOKEN: str

    # LLM
    LLM_BASE_URL: str = "https://router.huggingface.co/v1"
    HF_PROVIDER: str = "auto"

    MODEL_FAST: str = "meta-llama/Llama-3.1-8B-Instruct"
    MODEL_THINKING: str = "Qwen/Qwen2.5-7B-Instruct"

    LLM_DEFAULT_MODEL: str = "meta-llama/Llama-3.1-8B-Instruct"
    LLM_FALLBACK_MODEL: str = "Qwen/Qwen2.5-7B-Instruct"

    FAST_MAX_TOKENS: int = 1000
    THINKING_MAX_TOKENS: int = 2200

    FAST_TEMPERATURE: float = 0.35
    THINKING_TEMPERATURE: float = 0.25

    MODEL_REQUEST_TIMEOUT_SECONDS: int = 90
    MODEL_MAX_RETRIES: int = 2
    ALLOW_MODEL_FALLBACK: bool = True
    MODEL_HEALTHCHECK_ENABLED: bool = True

    # Backward compatibility
    MODEL_MEDIS_1: str = "Qwen/Qwen2.5-7B-Instruct"
    MODEL_MEDIS_2: str = "Qwen/Qwen2.5-7B-Instruct"
    MODEL_PELAJAR_1: str = "meta-llama/Llama-3.1-8B-Instruct"
    MODEL_PELAJAR_2: str = "Qwen/Qwen2.5-7B-Instruct"
    MODEL_UMUM: str = "meta-llama/Llama-3.1-8B-Instruct"

    GROQ_API_TOKEN: str = ""
    VLM_MODEL: str = "disabled"

    # OCR worker
    OCR_WORKER_URL: str = "http://medical_ocr_worker:8010"
    OCR_WORKER_TIMEOUT_SECONDS: int = 180
    OCR_WORKER_ENABLED: bool = True

    # GOT-OCR2
    OCR_MODEL_ID: str = "stepfun-ai/GOT-OCR-2.0-hf"
    OCR_MODEL_FALLBACK_ID: str = "stepfun-ai/GOT-OCR2_0"
    OCR_DEVICE: str = "auto"
    OCR_DTYPE: str = "auto"
    OCR_MAX_NEW_TOKENS: int = 2048
    OCR_TIMEOUT_SECONDS: int = 120
    OCR_LAZY_LOAD: bool = True
    OCR_TRUST_REMOTE_CODE: bool = False
    OCR_ENABLE_FALLBACK: bool = True
    OCR_MAX_IMAGE_PIXELS: int = 16_000_000
    OCR_CACHE_ENABLED: bool = True

    # Attachment
    ATTACHMENT_MAX_SIZE_MB: int = 20
    ATTACHMENT_CONTEXT_MAX_CHARS: int = 12_000
    ATTACHMENT_MAX_PDF_PAGES: int = 30
    ATTACHMENT_OCR_PAGE_LIMIT: int = 15

    # Verification
    NEO4J_ATTACHMENT_VERIFICATION: bool = True
    ATTACHMENT_MIN_CONFIDENCE: float = 0.55
    ATTACHMENT_HIGH_CONFIDENCE: float = 0.80

    EMBEDDING_MODEL_NAME: str = "intfloat/multilingual-e5-base"

    CORS_ORIGINS: str = "*"

    @field_validator("MINIO_ENDPOINT", mode="before")
    @classmethod
    def clean_minio_endpoint(cls, value: str) -> str:
        if isinstance(value, str):
            cleaned = value.replace("http://", "").replace("https://", "")
            return cleaned.split("/")[0].strip()

        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
