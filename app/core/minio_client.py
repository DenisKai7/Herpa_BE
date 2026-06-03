"""
MinIO Object Storage Client.

Reads credentials directly from os.getenv() — intentionally bypasses the
Pydantic settings layer to guarantee no intermediate transformation can
inject a protocol prefix into the endpoint string.

The MinIO Python SDK requires a raw 'hostname:port' endpoint.
Passing 'http://host:port' causes S3Error: Invalid Request (invalid hostname).
"""

import logging
import os

from minio import Minio

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════
# MINIO CLIENT (Object Storage)
# ═══════════════════════════════════════════

# Step 1: Read raw endpoint directly from environment
raw_endpoint = os.getenv("MINIO_ENDPOINT", "minio:9000")

# Step 2: Defensive parsing — strip protocol schemes and trailing paths
cleaned_endpoint = raw_endpoint.replace("http://", "").replace("https://", "")
cleaned_endpoint = cleaned_endpoint.split("/")[0].strip()

logger.info(f"MinIO endpoint sanitized: '{raw_endpoint}' -> '{cleaned_endpoint}'")

# Step 3: Instantiate global client with plain-HTTP enforcement
minio_client = Minio(
    cleaned_endpoint,
    access_key=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
    secret_key=os.getenv("MINIO_SECRET_KEY", "minioadmin"),
    secure=False,  # Force False — local Docker Compose runs plain HTTP
)

logger.info(
    f"MinIO client initialized: endpoint={cleaned_endpoint}, secure=False"
)


def ensure_bucket_exists(bucket_name: str = "chat-attachments") -> None:
    """Create bucket if it does not already exist."""
    try:
        if not minio_client.bucket_exists(bucket_name):
            minio_client.make_bucket(bucket_name)
            logger.info(f"MinIO bucket '{bucket_name}' created.")
        else:
            logger.info(f"MinIO bucket '{bucket_name}' already exists.")
    except Exception as e:
        logger.warning(f"MinIO bucket check failed (service may be down): {e}")
