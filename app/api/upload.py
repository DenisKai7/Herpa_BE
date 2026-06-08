"""
Upload API - attachment storage and multimodal extraction.

Files are stored in MinIO. Extracted attachment evidence is cached per user so
chat can reference it by attachment_id without exposing HF tokens or raw storage
credentials to the frontend.
"""

import io
import json
import logging
import uuid
from datetime import timedelta
from typing import Any, Optional

try:
    import redis
except Exception:
    redis = None
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.agent.multimodal import (
    ALL_ALLOWED_EXTENSIONS,
    ALLOWED_MIME_TYPES,
    build_attachment_context_package,
    format_attachment_context_package,
    process_attachment,
    sniff_mime_type,
    validate_file_type,
)
from app.core.config import settings
from app.core.dependencies import verify_user
from app.core.minio_client import minio_client
from app.models.schemas import UploadResponse

logger = logging.getLogger(__name__)
router = APIRouter()

ATTACHMENT_TTL_SECONDS = 24 * 60 * 60


def _redis_client():
    if redis is None:
        raise RuntimeError("Redis client unavailable")
    return redis.from_url(settings.REDIS_URL, decode_responses=True)


def _attachment_cache_key(user_id: str, attachment_id: str) -> str:
    return f"attachment:{user_id}:{attachment_id}"


def _public_preview_url(bucket: str, object_name: str) -> Optional[str]:
    try:
        url = minio_client.presigned_get_object(bucket, object_name, expires=timedelta(hours=1))
        public_endpoint = getattr(settings, "MINIO_PUBLIC_ENDPOINT", "").strip()
        if public_endpoint:
            from urllib.parse import urlparse, urlunparse
            parsed = urlparse(url)
            replacement = public_endpoint.replace("http://", "").replace("https://", "").rstrip("/")
            scheme = "https" if settings.MINIO_SECURE else "http"
            url = urlunparse((scheme, replacement, parsed.path, parsed.params, parsed.query, parsed.fragment))
        return url
    except Exception as exc:
        logger.warning("Failed to create MinIO preview URL: %s", exc)
        return None


def save_attachment_context(user_id: str, attachment_id: str, payload: dict[str, Any]) -> None:
    try:
        _redis_client().setex(_attachment_cache_key(user_id, attachment_id), ATTACHMENT_TTL_SECONDS, json.dumps(payload))
    except Exception as exc:
        logger.warning("Failed to cache attachment context: %s", exc)


def get_attachment_context_for_user(user_id: str, attachment_id: str) -> Optional[dict[str, Any]]:
    try:
        raw = _redis_client().get(_attachment_cache_key(user_id, attachment_id))
        return json.loads(raw) if raw else None
    except Exception as exc:
        logger.warning("Failed to read attachment context: %s", exc)
        return None


@router.post("/upload", response_model=UploadResponse, summary="Upload file untuk OCR")
async def upload_file(
    file: UploadFile = File(...),
    user_id: str = Depends(verify_user),
) -> UploadResponse:
    filename = file.filename or "unknown"
    declared_mime = (file.content_type or "").split(";")[0].strip().lower()

    try:
        file_data = await file.read()
        max_bytes = settings.ATTACHMENT_MAX_SIZE_MB * 1024 * 1024
        if len(file_data) > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"Ukuran file melebihi batas maksimal ({settings.ATTACHMENT_MAX_SIZE_MB} MB).",
            )

        ext = validate_file_type(filename, declared_mime, file_data)
        actual_mime = sniff_mime_type(file_data, filename)
        if actual_mime not in ALLOWED_MIME_TYPES or ext not in ALL_ALLOWED_EXTENSIONS:
            raise HTTPException(status_code=400, detail="unsupported_file")

        attachment_id = str(uuid.uuid4())
        unique_name = f"{attachment_id}.{ext}"
        bucket_name = settings.MINIO_BUCKET

        try:
            if not minio_client.bucket_exists(bucket_name):
                minio_client.make_bucket(bucket_name)
            minio_client.put_object(
                bucket_name,
                unique_name,
                io.BytesIO(file_data),
                len(file_data),
                content_type=actual_mime,
            )
        except Exception as minio_error:
            logger.error("MinIO upload failed: %s", type(minio_error).__name__, exc_info=True)
            raise HTTPException(status_code=503, detail="attachment_storage_failed")

        analysis = await process_attachment(
            filename=filename,
            mime_type=actual_mime,
            content=file_data,
            user_query=None,
        )
        analysis.attachment_id = attachment_id
        package = build_attachment_context_package(analysis)
        formatted_context = format_attachment_context_package(package)
        preview_url = _public_preview_url(bucket_name, unique_name)

        cache_payload = {
            "user_id": user_id,
            "attachment_id": attachment_id,
            "object_name": unique_name,
            "bucket": bucket_name,
            "filename": filename,
            "stored_filename": unique_name,
            "mime_type": actual_mime,
            "preview_url": preview_url,
            "analysis": analysis.model_dump(),
            "context_package": package.model_dump(),
            "formatted_context": formatted_context,
        }
        save_attachment_context(user_id, attachment_id, cache_payload)

        logger.info(
            "upload_completed user_id=%s attachment_id=%s filename=%s mime_type=%s file_sha256=%s processing_ms=%s verification_status=%s confidence=%.2f",
            user_id,
            attachment_id,
            filename,
            actual_mime,
            analysis.file_sha256,
            analysis.processing_ms,
            analysis.verification_status,
            analysis.confidence,
        )

        return UploadResponse(
            filename=unique_name,
            url=f"minio:{bucket_name}/{unique_name}",
            extracted_text=formatted_context,
            success=True,
            attachment={
                "id": attachment_id,
                "filename": filename,
                "stored_filename": unique_name,
                "mime_type": actual_mime,
                "preview_url": preview_url,
                "processing_status": "completed" if analysis.extracted_text or analysis.verification_status != "failed" else "failed",
                "detected_type": analysis.structured_data.get("detected_type", "unknown"),
                "verification_status": analysis.verification_status,
                "confidence": analysis.confidence,
            },
            context={
                "extracted_text": analysis.extracted_text,
                "summary": package.file_summary,
                "warnings": analysis.warnings,
            },
        )

    except HTTPException:
        raise
    except ValueError as exc:
        code = str(exc).split(":", 1)[0]
        status = 400 if code in {"corrupt_image", "unsupported_file"} else 422
        raise HTTPException(status_code=status, detail=code)
    except Exception as exc:
        logger.error("File upload error: %s", type(exc).__name__, exc_info=True)
        raise HTTPException(status_code=500, detail="attachment_processing_failed")

