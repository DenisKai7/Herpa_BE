"""
Embedding Service - HuggingFace Inference API.
Menyediakan fungsi encode teks ke embedding vector untuk pencarian semantik
menggunakan HuggingFace Inference API (cloud-based, tanpa download model lokal).

Model: sentence-transformers/paraphrase-multilingual-mpnet-base-v2 (768 dim, multilingual)
"""

import logging
import numpy as np
from huggingface_hub import InferenceClient
from app.core.config import settings

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════
# HUGGINGFACE INFERENCE CLIENT (Singleton)
# ═══════════════════════════════════════════
_client: InferenceClient | None = None
_embedding_dim: int | None = None


def _get_client() -> InferenceClient:
    """Lazy-init HuggingFace InferenceClient."""
    global _client
    if _client is None:
        logger.info(
            f"Initializing HuggingFace Inference Client for embedding model: "
            f"{settings.EMBEDDING_MODEL_NAME}"
        )
        _client = InferenceClient(
            provider="auto",
            api_key=settings.HF_API_TOKEN,
        )
        logger.info("HuggingFace Inference Client initialized successfully.")
    return _client


def embed_text(text: str) -> list[float]:
    """
    Mengubah satu teks menjadi embedding vector via HuggingFace Inference API.
    """
    client = _get_client()
    result = client.feature_extraction(
        text=text,
        model=settings.EMBEDDING_MODEL_NAME,
    )
    embedding = np.array(result)
    # Jika hasilnya 2D (token-level), lakukan mean pooling
    if embedding.ndim == 2:
        embedding = embedding.mean(axis=0)
    # Jika hasilnya 3D, ambil pertama lalu mean pool
    if embedding.ndim == 3:
        embedding = embedding[0].mean(axis=0)
    # L2 Normalize
    norm = np.linalg.norm(embedding)
    embedding = embedding / (norm if norm > 0 else 1.0)
    return embedding.tolist()


def embed_texts(texts: list[str], is_passage: bool = False) -> list[list[float]]:
    """
    Mengubah banyak teks menjadi embedding vectors via HuggingFace Inference API.

    Args:
        texts: List of text strings.
        is_passage: Flag untuk kompatibilitas (tidak digunakan oleh model ini).
    """
    client = _get_client()
    all_embeddings = []
    for text in texts:
        result = client.feature_extraction(
            text=text,
            model=settings.EMBEDDING_MODEL_NAME,
        )
        embedding = np.array(result)
        if embedding.ndim == 2:
            embedding = embedding.mean(axis=0)
        if embedding.ndim == 3:
            embedding = embedding[0].mean(axis=0)
        norm = np.linalg.norm(embedding)
        embedding = embedding / (norm if norm > 0 else 1.0)
        all_embeddings.append(embedding.tolist())
    return all_embeddings


def get_embedding_dimension() -> int:
    """
    Mengembalikan dimensi embedding model yang aktif.
    Dilakukan dengan mengirim teks dummy untuk mendeteksi dimensi.
    """
    global _embedding_dim
    if _embedding_dim is None:
        test_embedding = embed_text("test")
        _embedding_dim = len(test_embedding)
        logger.info(f"Detected embedding dimension: {_embedding_dim}")
    return _embedding_dim
