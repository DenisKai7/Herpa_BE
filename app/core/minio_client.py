"""
MinIO Object Storage Client.
Menyediakan koneksi ke MinIO untuk upload file attachment chat.
"""

import logging
from minio import Minio
from app.core.config import settings

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════
# MINIO CLIENT (Object Storage)
# ═══════════════════════════════════════════
minio_client = Minio(
    endpoint=settings.MINIO_ENDPOINT,
    access_key=settings.MINIO_ACCESS_KEY,
    secret_key=settings.MINIO_SECRET_KEY,
    secure=settings.MINIO_SECURE,
)


def ensure_bucket_exists():
    """Membuat bucket jika belum ada."""
    bucket_name = settings.MINIO_BUCKET
    try:
        if not minio_client.bucket_exists(bucket_name):
            minio_client.make_bucket(bucket_name)
            logger.info(f"MinIO bucket '{bucket_name}' created successfully.")
        else:
            logger.info(f"MinIO bucket '{bucket_name}' already exists.")
    except Exception as e:
        logger.warning(f"MinIO bucket check failed (service may be down): {e}")
