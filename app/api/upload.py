"""
Upload API - Multimodal file processing (PDF, Image OCR, TXT).

Files di-upload ke MinIO dan teks di-ekstrak untuk dipakai sebagai
file_context oleh AI agent dalam pipeline chat.

Keamanan:
- Validasi MIME type ketat (hanya application/pdf & image standard).
- Validasi ekstensi file.
- Batas ukuran file maksimal 10 MB.
- Endpoint dilindungi oleh JWT verification.
"""

import io
import logging
import uuid

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.agent.multimodal import (
    ALL_ALLOWED_EXTENSIONS,
    ALLOWED_MIME_TYPES,
    extract_text_from_file,
)
from app.core.config import settings
from app.core.dependencies import verify_user
from app.core.minio_client import minio_client
from app.models.schemas import UploadResponse

logger = logging.getLogger(__name__)
router = APIRouter()

# Konstanta
MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10 MB


@router.post(
    "/upload",
    response_model=UploadResponse,
    summary="Upload file untuk OCR",
)
async def upload_file(
    file: UploadFile = File(...),
    user_id: str = Depends(verify_user),
) -> UploadResponse:
    """
    Upload file (PDF/Image/TXT) ke MinIO dan ekstrak teksnya.

    Pipeline:
    1. Validasi MIME type (Security: hanya tipe yang diizinkan).
    2. Validasi ekstensi file.
    3. Validasi ukuran file (maks 10 MB).
    4. Baca bytes file.
    5. Ekstrak teks (PyMuPDF untuk PDF, Tesseract untuk Image, decode untuk TXT).
    6. Upload file asli ke MinIO bucket.
    7. Return URL MinIO + extracted text.

    Extracted text kemudian dikirim sebagai `file_context` di ChatRequest.

    Args:
        file: File upload dari multipart form.
        user_id: UUID user dari JWT (injected by Depends).

    Returns:
        UploadResponse berisi filename, URL MinIO, dan teks yang diekstrak.
    """
    filename = file.filename or "unknown"

    # ── Validasi MIME type (SECURITY FIRST per brief) ──
    content_type = file.content_type or ""
    normalized_mime = content_type.split(";")[0].strip().lower()

    if normalized_mime not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"MIME type '{normalized_mime}' tidak diizinkan. "
                f"Tipe yang diterima: {', '.join(sorted(ALLOWED_MIME_TYPES.keys()))}"
            ),
        )

    # ── Validasi ekstensi ──
    ext = filename.rsplit(".", maxsplit=1)[-1].lower() if "." in filename else ""
    if ext not in ALL_ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Format file '.{ext}' tidak didukung. "
                f"Format yang diizinkan: {', '.join(sorted(ALL_ALLOWED_EXTENSIONS))}"
            ),
        )

    # Cross-check MIME type vs extension
    allowed_exts_for_mime = ALLOWED_MIME_TYPES.get(normalized_mime, frozenset())
    if ext not in allowed_exts_for_mime:
        raise HTTPException(
            status_code=400,
            detail=f"MIME type '{normalized_mime}' tidak cocok dengan ekstensi '.{ext}'.",
        )

    try:
        # ── Baca file ──
        file_data = await file.read()

        # ── Validasi ukuran ──
        if len(file_data) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"Ukuran file melebihi batas maksimal "
                    f"({MAX_FILE_SIZE // (1024 * 1024)} MB)."
                ),
            )

        # ── Ekstrak teks ──
        extracted_content = extract_text_from_file(
            file_bytes=file_data,
            filename=filename,
            content_type=normalized_mime,
        )
        logger.info(
            f"Extracted {len(extracted_content)} chars from '{filename}' "
            f"(user: {user_id})"
        )

        # ── Upload ke MinIO ──
        unique_name = f"{uuid.uuid4()}.{ext}"
        bucket_name = "chat-attachments"
        if not minio_client.bucket_exists(bucket_name):
            minio_client.make_bucket(bucket_name)

        minio_client.put_object(
            bucket_name,
            unique_name,
            io.BytesIO(file_data),
            len(file_data),
            content_type=content_type or "application/octet-stream",
        )
        logger.info(f"File uploaded to MinIO: {bucket_name}/{unique_name}")

        return UploadResponse(
            filename=unique_name,
            url=f"minio:{bucket_name}/{unique_name}",
            extracted_text=extracted_content,
        )

    except HTTPException:
        raise
    except ValueError as e:
        # Dari multimodal.validate_file_type
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        # Dari multimodal extraction errors
        logger.error(f"File extraction error: {e}", exc_info=True)
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"File upload error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Gagal memproses file: {str(e)}",
        )
