"""HTTP client for the internal OCR worker."""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx

from app.agent.multimodal import (
    AttachmentAnalysisResult,
    OcrExtractionResult,
    _analysis_from_ocr_result,
    calculate_sha256,
    classify_extracted_text,
)
from app.core.config import settings

logger = logging.getLogger(__name__)


class OcrWorkerClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    async def extract(
        self,
        *,
        filename: str,
        mime_type: str,
        content: bytes,
    ) -> dict[str, Any]:
        timeout = httpx.Timeout(float(settings.OCR_WORKER_TIMEOUT_SECONDS))
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                f"{self.base_url}/internal/ocr/extract",
                files={"file": (filename, content, mime_type)},
            )
            response.raise_for_status()
            return response.json()


async def extract_attachment_with_worker(
    *,
    filename: str,
    mime_type: str,
    content: bytes,
    user_query: str | None = None,
) -> AttachmentAnalysisResult:
    start = time.time()
    if not settings.OCR_WORKER_ENABLED:
        return _degraded_result(filename, mime_type, content, start, ["ocr_worker_disabled"])

    client = OcrWorkerClient(settings.OCR_WORKER_URL)
    try:
        payload = await client.extract(filename=filename, mime_type=mime_type, content=content)
    except httpx.HTTPStatusError as exc:
        logger.warning("OCR worker returned HTTP %s", exc.response.status_code)
        return _degraded_result(filename, mime_type, content, start, [f"ocr_worker_http_{exc.response.status_code}"])
    except (httpx.ConnectError, httpx.ReadTimeout, httpx.ConnectTimeout, httpx.RemoteProtocolError) as exc:
        logger.warning("OCR worker unavailable: %s", type(exc).__name__)
        return _degraded_result(filename, mime_type, content, start, ["ocr_worker_unavailable", type(exc).__name__])

    result = OcrExtractionResult(
        success=bool(payload.get("success", True)),
        raw_text=str(payload.get("raw_text") or payload.get("extracted_text") or ""),
        normalized_text=str(payload.get("normalized_text") or ""),
        detected_type=str(payload.get("detected_type") or "unknown"),
        visible_labels=list(payload.get("visible_labels") or []),
        chemical_terms=list(payload.get("chemical_terms") or []),
        molecular_formulas=list(payload.get("molecular_formulas") or []),
        numeric_labels=list(payload.get("numeric_labels") or []),
        document_sections=list(payload.get("document_sections") or []),
        tables=list(payload.get("tables") or []),
        confidence=float(payload.get("confidence") or 0.0),
        warnings=list(payload.get("warnings") or []),
        model_id=str(payload.get("model_id") or "ocr-worker"),
        processing_ms=int(payload.get("processing_ms") or int((time.time() - start) * 1000)),
    )
    method = str(payload.get("method") or "ocr-worker")
    return _analysis_from_ocr_result(
        filename=filename,
        mime_type=mime_type,
        content=content,
        result=result,
        method=method,
    )


def _degraded_result(
    filename: str,
    mime_type: str,
    content: bytes,
    start: float,
    warnings: list[str],
) -> AttachmentAnalysisResult:
    text = ""
    confidence = 0.0
    detected_type = "unknown"
    if mime_type in {"text/plain", "application/json"}:
        for encoding in ("utf-8", "latin-1"):
            try:
                text = content.decode(encoding)
                confidence = 1.0 if text.strip() else 0.0
                detected_type = "plain_text"
                break
            except UnicodeDecodeError:
                continue

    heuristics = classify_extracted_text(text)
    return AttachmentAnalysisResult(
        filename=filename,
        mime_type=mime_type,
        file_sha256=calculate_sha256(content),
        extraction_method="ocr-worker-degraded",
        extracted_text=text[: settings.ATTACHMENT_CONTEXT_MAX_CHARS],
        structured_data={**heuristics, "detected_type": detected_type},
        verification_status="not_applicable" if text else "failed",
        confidence=confidence,
        warnings=warnings,
        processing_ms=int((time.time() - start) * 1000),
    )
