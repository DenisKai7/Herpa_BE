"""
Upload API - attachment storage and multimodal extraction.

Files are stored in MinIO. Extracted attachment evidence is cached per user so
chat can reference it by attachment_id without exposing HF tokens or raw storage
credentials to the frontend.
"""

import io
import json
import logging
import time
import uuid
from datetime import datetime, timezone, timedelta
from typing import Any, Optional

try:
    import redis
except Exception:
    redis = None
from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile

from app.agent.multimodal import (
    ALL_ALLOWED_EXTENSIONS,
    ALLOWED_MIME_TYPES,
    VisualExtractionResult,
    build_attachment_context_package,
    format_attachment_context_package,
    process_attachment,
    sniff_mime_type,
    validate_file_type,
)
from app.agent.verification import verify_attachment_with_neo4j
from app.core.config import settings
from uuid import UUID
from app.api.auth import get_current_user
from app.core.dependencies import verify_user
from app.core.minio_client import minio_client
from app.models.schemas import UploadResponse, AttachmentRetryResponse

logger = logging.getLogger(__name__)
router = APIRouter()

ATTACHMENT_TTL_SECONDS = settings.ATTACHMENT_STATUS_TTL_SECONDS
MAX_ATTACHMENT_RETRIES = 3


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _redis_client():
    if redis is None:
        raise RuntimeError("Redis client unavailable")
    return redis.from_url(settings.REDIS_URL, decode_responses=True)


def _attachment_cache_key(user_id: str, attachment_id: str) -> str:
    return f"attachment:{user_id}:{attachment_id}"


def _processing_cache_key(attachment_id: str) -> str:
    return f"attachment:processing:{attachment_id}"


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
        client = _redis_client()
        encoded = json.dumps(payload)
        client.setex(_attachment_cache_key(user_id, attachment_id), ATTACHMENT_TTL_SECONDS, encoded)
        client.setex(_processing_cache_key(attachment_id), ATTACHMENT_TTL_SECONDS, encoded)
    except Exception as exc:
        logger.warning("Failed to cache attachment context: %s", exc)


def get_attachment_context_for_user(user_id: str, attachment_id: str) -> Optional[dict[str, Any]]:
    try:
        raw = _redis_client().get(_attachment_cache_key(user_id, attachment_id))
        return json.loads(raw) if raw else None
    except Exception as exc:
        logger.warning("Failed to read attachment context: %s", exc)
        return None


def _get_owned_attachment_or_404(user_id: str, attachment_id: str) -> dict[str, Any]:
    payload = get_attachment_context_for_user(user_id, attachment_id)
    if not payload:
        raise HTTPException(status_code=404, detail="Attachment tidak ditemukan atau bukan milik user ini.")
    if payload.get("user_id") != user_id:
        raise HTTPException(status_code=403, detail="Anda tidak memiliki akses ke attachment ini.")
    return payload


def _update_attachment_payload(user_id: str, attachment_id: str, updates: dict[str, Any]) -> dict[str, Any]:
    payload = get_attachment_context_for_user(user_id, attachment_id) or {"user_id": user_id, "attachment_id": attachment_id}
    payload.update(updates)
    payload["updated_at"] = _now_iso()
    save_attachment_context(user_id, attachment_id, payload)
    return payload


def _attachment_response(payload: dict[str, Any]) -> dict[str, Any]:
    processing_status = payload.get("processing_status", "queued")
    response: dict[str, Any] = {
        "attachment_id": payload.get("attachment_id"),
        "processing_status": processing_status,
        "progress": int(payload.get("progress") or 0),
        "verification_status": payload.get("verification_status", "pending"),
        "confidence": float(payload.get("confidence") or 0.0),
        "retryable": bool(payload.get("retryable", processing_status == "failed")),
    }
    if processing_status == "completed":
        analysis = payload.get("analysis") or {}
        response["extracted_text"] = analysis.get("extracted_text") or ""
        response["detected_type"] = (analysis.get("structured_data") or {}).get("detected_type", "unknown")
    if processing_status == "failed":
        response["error"] = payload.get("error") or {
            "code": "VLM_PROCESSING_FAILED",
            "message": "Gambar belum berhasil dianalisis.",
        }
    return response


def _read_minio_object(bucket: str, object_name: str) -> bytes:
    response = minio_client.get_object(bucket, object_name)
    try:
        return response.read()
    finally:
        response.close()
        response.release_conn()


async def _verify_analysis_if_available(analysis, user_query: str | None = None):
    if not analysis.extracted_text.strip():
        return analysis
    extraction = VisualExtractionResult(
        success=True,
        raw_text=analysis.extracted_text,
        normalized_text=analysis.extracted_text,
        detected_type=analysis.structured_data.get("detected_type", "unknown"),
        visible_labels=list(analysis.structured_data.get("visible_labels") or analysis.structured_data.get("chemical_symbols") or []),
        chemical_terms=list(analysis.structured_data.get("chemical_terms") or []),
        molecular_formulas=list(analysis.structured_data.get("molecular_formulas") or []),
        numeric_labels=list(analysis.structured_data.get("numeric_labels") or []),
        document_sections=list(analysis.structured_data.get("document_sections") or []),
        tables=list(analysis.structured_data.get("tables") or []),
        confidence=analysis.confidence,
        warnings=analysis.warnings,
        model_id=analysis.extraction_method,
        processing_ms=analysis.processing_ms,
    )
    verification = await verify_attachment_with_neo4j(extraction, user_query or "")
    analysis.neo4j_candidates = [candidate.model_dump() for candidate in verification.candidates]
    analysis.warnings.extend(verification.warnings or [])
    analysis.structured_data["limitations"] = list(analysis.structured_data.get("limitations") or []) + list(verification.limitations or [])
    if verification.verification_status == "failed" and not verification.success:
        analysis.verification_status = "unavailable"
    else:
        analysis.verification_status = verification.verification_status
        if verification.confidence:
            analysis.confidence = verification.confidence
    return analysis


async def _process_attachment_job(
    *,
    user_id: str,
    attachment_id: str,
    filename: str,
    mime_type: str,
    content: bytes,
) -> None:
    started = time.perf_counter()
    vlm_job_id = str(uuid.uuid4())
    _update_attachment_payload(user_id, attachment_id, {"processing_status": "processing", "progress": 40, "error": None, "retryable": False})
    try:
        analysis = await process_attachment(filename=filename, mime_type=mime_type, content=content, user_query=None)
        analysis.attachment_id = attachment_id
        if analysis.extracted_text.strip():
            analysis = await _verify_analysis_if_available(analysis)
        package = build_attachment_context_package(analysis)
        formatted_context = format_attachment_context_package(package)
        status = "completed" if analysis.extracted_text or analysis.verification_status != "failed" else "failed"
        updates: dict[str, Any] = {
            "processing_status": status,
            "progress": 100,
            "verification_status": analysis.verification_status,
            "confidence": analysis.confidence,
            "analysis": analysis.model_dump(),
            "context_package": package.model_dump(),
            "formatted_context": formatted_context,
            "retryable": status == "failed",
        }
        if status == "failed":
            updates["error"] = {"code": "VLM_PROCESSING_FAILED", "message": "Gambar belum berhasil dianalisis."}
        _update_attachment_payload(user_id, attachment_id, updates)
        logger.info(
            "vlm_job_completed user_id=%s attachment_id=%s vlm_job_id=%s processing_ms=%s verification_status=%s",
            user_id,
            attachment_id,
            vlm_job_id,
            int((time.perf_counter() - started) * 1000),
            analysis.verification_status,
        )
    except Exception as exc:
        logger.exception(
            "vlm_job_failed user_id=%s attachment_id=%s vlm_job_id=%s error_type=%s processing_ms=%s",
            user_id,
            attachment_id,
            vlm_job_id,
            type(exc).__name__,
            int((time.perf_counter() - started) * 1000),
        )
        _update_attachment_payload(
            user_id,
            attachment_id,
            {
                "processing_status": "failed",
                "progress": 100,
                "verification_status": "failed",
                "confidence": 0.0,
                "retryable": True,
                "error": {"code": "VLM_PROCESSING_FAILED", "message": "Gambar belum berhasil dianalisis."},
            },
        )


@router.post("/upload", response_model=UploadResponse, summary="Upload file untuk analisis attachment")
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    user_id: str = Depends(verify_user),
) -> UploadResponse:
    filename = file.filename or "unknown"
    declared_mime = (file.content_type or "").split(";")[0].strip().lower()
    upload_started = time.perf_counter()

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
            "processing_status": "queued",
            "progress": 0,
            "verification_status": "pending",
            "confidence": 0.0,
            "retry_count": 0,
            "retryable": False,
            "analysis": None,
            "context_package": None,
            "formatted_context": "",
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
        }
        save_attachment_context(user_id, attachment_id, cache_payload)
        background_tasks.add_task(
            _process_attachment_job,
            user_id=user_id,
            attachment_id=attachment_id,
            filename=filename,
            mime_type=actual_mime,
            content=file_data,
        )

        logger.info(
            "upload_queued user_id=%s attachment_id=%s filename=%s mime_type=%s file_size=%s processing_ms=%s verification_status=%s confidence=%.2f",
            user_id,
            attachment_id,
            filename,
            actual_mime,
            len(file_data),
            int((time.perf_counter() - upload_started) * 1000),
            "pending",
            0.0,
        )

        return UploadResponse(
            filename=unique_name,
            url=f"minio:{bucket_name}/{unique_name}",
            extracted_text="",
            success=True,
            attachment={
                "id": attachment_id,
                "filename": filename,
                "stored_filename": unique_name,
                "mime_type": actual_mime,
                "preview_url": preview_url,
                "processing_status": "queued",
                "detected_type": "unknown",
                "verification_status": "pending",
                "confidence": 0.0,
            },
            context={"extracted_text": "", "summary": "", "warnings": []},
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


@router.get("/{attachment_id}/status", summary="Status pemrosesan attachment")
async def get_attachment_status(
    attachment_id: str,
    user_id: str = Depends(verify_user),
) -> dict[str, Any]:
    payload = _get_owned_attachment_or_404(user_id, attachment_id)
    return _attachment_response(payload)


@router.post(
    "/{attachment_id}/retry",
    response_model=AttachmentRetryResponse,
    summary="Ulangi analisis attachment tanpa upload ulang",
)
async def retry_attachment(
    attachment_id: UUID,
    background_tasks: BackgroundTasks,
    current_user=Depends(get_current_user),
) -> AttachmentRetryResponse:
    user_id = current_user.get("id")
    if not user_id:
        raise HTTPException(status_code=401, detail="User tidak teridentifikasi.")

    attachment_id_str = str(attachment_id)
    payload = _get_owned_attachment_or_404(user_id, attachment_id_str)
    retry_count = int(payload.get("retry_count") or 0)
    if retry_count >= MAX_ATTACHMENT_RETRIES:
        raise HTTPException(
            status_code=429,
            detail={
                "code": "ATTACHMENT_RETRY_LIMIT_REACHED",
                "message": "Batas percobaan ulang analisis attachment telah tercapai.",
                "retryable": False,
            },
        )

    try:
        content = _read_minio_object(payload["bucket"], payload["object_name"])
    except Exception as exc:
        logger.error("Attachment retry MinIO read failed attachment_id=%s error_type=%s", attachment_id_str, type(exc).__name__, exc_info=True)
        raise HTTPException(status_code=503, detail="attachment_storage_read_failed") from exc

    payload = _update_attachment_payload(
        user_id,
        attachment_id_str,
        {
            "processing_status": "queued",
            "progress": 0,
            "verification_status": "pending",
            "confidence": 0.0,
            "retry_count": retry_count + 1,
            "retryable": False,
            "error": None,
        },
    )
    background_tasks.add_task(
        _process_attachment_job,
        user_id=user_id,
        attachment_id=attachment_id_str,
        filename=payload["filename"],
        mime_type=payload["mime_type"],
        content=content,
    )
    response_data = _attachment_response(payload)
    return AttachmentRetryResponse.model_validate(response_data)
