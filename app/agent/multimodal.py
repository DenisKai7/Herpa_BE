"""
Multimodal attachment helpers for backend-safe upload processing.

Heavy OCR/ML dependencies live in app.ocr_worker. This module must stay importable
inside the main backend image without torch, transformers, PyMuPDF, Pillow, or pandas.
"""

from __future__ import annotations

import hashlib
import io
import logging
import re
import time
from typing import Any, Final, Optional

from pydantic import BaseModel, Field

from app.core.config import settings

logger = logging.getLogger(__name__)

PDF_EXTENSIONS: Final[frozenset[str]] = frozenset({"pdf"})
IMAGE_EXTENSIONS: Final[frozenset[str]] = frozenset({"jpg", "jpeg", "png", "webp", "bmp", "tiff"})
TEXT_EXTENSIONS: Final[frozenset[str]] = frozenset({"txt", "md", "json"})
DOCX_EXTENSIONS: Final[frozenset[str]] = frozenset({"docx"})
EXCEL_CSV_EXTENSIONS: Final[frozenset[str]] = frozenset({"xlsx", "xls", "csv"})

ALL_ALLOWED_EXTENSIONS: Final[frozenset[str]] = (
    PDF_EXTENSIONS | IMAGE_EXTENSIONS | TEXT_EXTENSIONS | DOCX_EXTENSIONS | EXCEL_CSV_EXTENSIONS
)

ALLOWED_MIME_TYPES: Final[dict[str, frozenset[str]]] = {
    "application/pdf": PDF_EXTENSIONS,
    "image/jpeg": frozenset({"jpg", "jpeg"}),
    "image/png": frozenset({"png"}),
    "image/webp": frozenset({"webp"}),
    "image/bmp": frozenset({"bmp"}),
    "image/tiff": frozenset({"tiff"}),
    "text/plain": frozenset({"txt", "md"}),
    "application/json": frozenset({"json"}),
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": DOCX_EXTENSIONS,
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": frozenset({"xlsx"}),
    "application/vnd.ms-excel": frozenset({"xls"}),
    "text/csv": frozenset({"csv"}),
}


class OcrExtractionResult(BaseModel):
    success: bool
    raw_text: str = ""
    normalized_text: str = ""
    detected_type: str = "unknown"
    visible_labels: list[str] = Field(default_factory=list)
    chemical_terms: list[str] = Field(default_factory=list)
    molecular_formulas: list[str] = Field(default_factory=list)
    numeric_labels: list[str] = Field(default_factory=list)
    document_sections: list[str] = Field(default_factory=list)
    tables: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    warnings: list[str] = Field(default_factory=list)
    model_id: str = "ocr-worker"
    processing_ms: int = 0


class AttachmentAnalysisResult(BaseModel):
    attachment_id: Optional[str] = None
    filename: str
    mime_type: str
    file_sha256: str
    extraction_method: str
    extracted_text: str
    structured_data: dict = Field(default_factory=dict)
    detected_entities: list[dict] = Field(default_factory=list)
    neo4j_candidates: list[dict] = Field(default_factory=list)
    verification_status: str = "unverified"
    confidence: float = 0.0
    warnings: list[str] = Field(default_factory=list)
    processing_ms: int


class _LegacyOcrService:
    async def extract(self, image, *, mode: str = "auto") -> OcrExtractionResult:
        raise RuntimeError("ocr_worker_required")


ocr_service = _LegacyOcrService()


def validate_file_type(filename: str, content_type: str | None = None, content: bytes | None = None) -> str:
    if "." not in filename:
        raise ValueError(f"File '{filename}' tidak memiliki ekstensi.")

    ext = filename.rsplit(".", maxsplit=1)[-1].lower()
    if ext not in ALL_ALLOWED_EXTENSIONS:
        raise ValueError(
            f"Format file '.{ext}' tidak didukung. Format yang diizinkan: {', '.join(sorted(ALL_ALLOWED_EXTENSIONS))}"
        )

    if content is not None:
        sniffed_mime = sniff_mime_type(content, filename)
        if sniffed_mime not in ALLOWED_MIME_TYPES:
            raise ValueError(f"MIME type dari isi file '{sniffed_mime}' tidak diizinkan.")
        if ext not in ALLOWED_MIME_TYPES[sniffed_mime]:
            raise ValueError(f"MIME type dari isi file '{sniffed_mime}' tidak cocok dengan ekstensi '.{ext}'.")

    if content_type:
        normalized_mime = content_type.split(";")[0].strip().lower()
        if normalized_mime not in ALLOWED_MIME_TYPES:
            raise ValueError(
                f"MIME type '{normalized_mime}' tidak diizinkan. Tipe yang diterima: {', '.join(sorted(ALLOWED_MIME_TYPES.keys()))}"
            )
        if ext not in ALLOWED_MIME_TYPES[normalized_mime]:
            raise ValueError(f"MIME type '{normalized_mime}' tidak cocok dengan ekstensi '.{ext}'.")

    return ext


def sniff_mime_type(content: bytes, filename: str = "") -> str:
    head = content[:4096]
    if head.startswith(b"%PDF"):
        return "application/pdf"
    if head.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if head.startswith(b"RIFF") and b"WEBP" in head[:16]:
        return "image/webp"
    if head.startswith(b"BM"):
        return "image/bmp"
    if head.startswith((b"II*\x00", b"MM\x00*")):
        return "image/tiff"
    if content[:4] == b"PK\x03\x04":
        import zipfile

        try:
            with zipfile.ZipFile(io.BytesIO(content)) as zf:
                names = set(zf.namelist())
                if "word/document.xml" in names:
                    return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                if "xl/workbook.xml" in names:
                    return "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        except Exception:
            pass
    try:
        content[:8192].decode("utf-8")
        return "text/plain"
    except UnicodeDecodeError:
        return "application/octet-stream"


def preprocess_image(data: bytes):
    try:
        from PIL import Image, ImageOps
    except Exception as exc:
        raise RuntimeError("Pillow tidak tersedia di backend utama; gunakan OCR worker.") from exc

    Image.MAX_IMAGE_PIXELS = settings.OCR_MAX_IMAGE_PIXELS
    try:
        with Image.open(io.BytesIO(data)) as verifier:
            verifier.verify()
        image = Image.open(io.BytesIO(data))
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")
    except Exception as exc:
        raise ValueError("corrupt_image") from exc

    if image.width * image.height > settings.OCR_MAX_IMAGE_PIXELS:
        logger.info("Image pixels exceed limit (%sx%s), resizing", image.width, image.height)
        image.thumbnail((4096, 4096))
    return image


def calculate_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _extract_from_text(file_bytes: bytes) -> str:
    try:
        return file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return file_bytes.decode("latin-1")


def classify_extracted_text(text: str) -> dict[str, Any]:
    word_count = len(text.split())
    chemical_symbols = list(set(re.findall(r"\b(OH|COOH|NH2|CH3|OCH3|CH2|C\d+H\d+|H2O|CO2)\b", text)))
    molecular_formulas = list(set(re.findall(r"\b[CNO SPHFIBrcl]{1,8}\d+[A-Za-z0-9]*\b", text)))
    molecular_formulas = [f for f in molecular_formulas if any(c.isdigit() for c in f) and len(f) > 2]
    chemical_terms = list(set(re.findall(r"\b(flavonoid|alkaloid|saponin|tannin|curcumin|gingerol|xanthorrhizol|terpenoid|steroid|glycoside|phenolic)\b", text.lower())))
    numeric_labels = list(set(re.findall(r"\b\d{1,2}\b", text)))
    has_table = "|" in text and text.count("|") > 4
    tables = [line for line in text.splitlines() if "|" in line] if has_table else []
    sections = [line.strip() for line in text.splitlines() if re.match(r"^(bab|chapter|section|abstrak|abstract|daftar|tabel|gambar|\d+\.)", line.lower().strip())]

    detected_type = "unknown"
    if has_table or any(k in text.lower() for k in ["table", "tabel"]):
        detected_type = "table"
    elif len(chemical_symbols) > 1 or molecular_formulas or any(k in text.lower() for k in ["benzene", "cincin", "skeletal", "structure"]):
        detected_type = "chemical_structure_diagram"
    elif any(k in text.lower() for k in ["equation", "persamaan", "formula"]) and any(s in text for s in ["+", "=", "-", "*", "/"]):
        detected_type = "mathematical_formula"
    elif word_count > 100:
        detected_type = "scanned_document"
    elif word_count > 0:
        detected_type = "plain_text"

    return {
        "detected_type": detected_type,
        "chemical_symbols": chemical_symbols,
        "molecular_formulas": molecular_formulas,
        "chemical_terms": chemical_terms,
        "numeric_labels": numeric_labels,
        "document_sections": sections,
        "tables": tables,
    }


def _analysis_from_ocr_result(*, filename: str, mime_type: str, content: bytes, result: OcrExtractionResult, method: str) -> AttachmentAnalysisResult:
    heuristics = classify_extracted_text(result.raw_text)
    structured_data = {
        "visible_labels": result.visible_labels or heuristics["chemical_symbols"],
        "chemical_terms": result.chemical_terms or heuristics["chemical_terms"],
        "molecular_formulas": result.molecular_formulas or heuristics["molecular_formulas"],
        "numeric_labels": result.numeric_labels or heuristics["numeric_labels"],
        "document_sections": result.document_sections or heuristics["document_sections"],
        "tables": result.tables or heuristics["tables"],
        "detected_type": result.detected_type if result.detected_type != "unknown" else heuristics["detected_type"],
    }
    return AttachmentAnalysisResult(
        filename=filename,
        mime_type=mime_type,
        file_sha256=calculate_sha256(content),
        extraction_method=method,
        extracted_text=result.raw_text[: settings.ATTACHMENT_CONTEXT_MAX_CHARS],
        structured_data=structured_data,
        detected_entities=[{"type": "compound_term", "value": value} for value in structured_data["chemical_terms"]]
        + [{"type": "molecular_formula", "value": value} for value in structured_data["molecular_formulas"]],
        verification_status="not_applicable",
        confidence=result.confidence,
        warnings=result.warnings,
        processing_ms=result.processing_ms,
    )


async def process_attachment(
    *,
    filename: str,
    mime_type: str,
    content: bytes,
    user_query: str | None,
) -> AttachmentAnalysisResult:
    start_time = time.time()
    ext = validate_file_type(filename, mime_type, content)

    if ext in TEXT_EXTENSIONS:
        text = _extract_from_text(content)
        heuristics = classify_extracted_text(text)
        return AttachmentAnalysisResult(
            filename=filename,
            mime_type=mime_type,
            file_sha256=calculate_sha256(content),
            extraction_method="text-layer",
            extracted_text=text[: settings.ATTACHMENT_CONTEXT_MAX_CHARS],
            structured_data={**heuristics, "detected_type": "plain_text"},
            verification_status="not_applicable",
            confidence=1.0 if text.strip() else 0.0,
            processing_ms=int((time.time() - start_time) * 1000),
        )

    patched_extract = getattr(ocr_service, "extract", None)
    if ext in IMAGE_EXTENSIONS and callable(patched_extract) and getattr(patched_extract, "__self__", None) is None:
        image = preprocess_image(content)
        ocr_result = await patched_extract(image)
        return _analysis_from_ocr_result(
            filename=filename,
            mime_type=mime_type,
            content=content,
            result=ocr_result,
            method="GOT-OCR2",
        )

    try:
        from app.core.ocr_client import extract_attachment_with_worker

        return await extract_attachment_with_worker(
            filename=filename,
            mime_type=mime_type,
            content=content,
            user_query=user_query,
        )
    except Exception as exc:
        logger.warning("OCR worker unavailable or failed: %s", exc)
        return AttachmentAnalysisResult(
            filename=filename,
            mime_type=mime_type,
            file_sha256=calculate_sha256(content),
            extraction_method="ocr-worker-unavailable",
            extracted_text="",
            structured_data={"detected_type": "unknown"},
            verification_status="failed",
            confidence=0.0,
            warnings=["ocr_worker_unavailable", str(exc)[:200]],
            processing_ms=int((time.time() - start_time) * 1000),
        )


def extract_text_from_file(file_bytes: bytes, filename: str, content_type: str | None = None) -> str:
    ext = validate_file_type(filename, content_type, file_bytes)
    if ext in TEXT_EXTENSIONS:
        return _extract_from_text(file_bytes)[: settings.ATTACHMENT_CONTEXT_MAX_CHARS]
    raise RuntimeError("Ekstraksi file non-teks dipindahkan ke OCR worker.")


class AttachmentContextPackage(BaseModel):
    user_question: str = ""
    file_summary: str
    ocr_text: str
    detected_type: str
    extracted_entities: list[dict] = Field(default_factory=list)
    verified_candidates: list[dict] = Field(default_factory=list)
    neo4j_evidence: list[dict] = Field(default_factory=list)
    confidence: float = 0.0
    verification_status: str = "unverified"
    limitations: list[str] = Field(default_factory=list)


def build_attachment_context_package(
    analysis: AttachmentAnalysisResult,
    *,
    user_question: str = "",
) -> AttachmentContextPackage:
    detected_type = analysis.structured_data.get("detected_type", "unknown")
    limitations = list(analysis.structured_data.get("limitations") or [])
    limitations.extend(analysis.warnings or [])
    if analysis.verification_status not in {"verified", "not_applicable"}:
        limitations.append("Identitas molekul atau tanaman belum boleh dianggap pasti karena evidence verifikasi belum kuat.")
    if detected_type == "chemical_structure_diagram":
        limitations.append("GOT-OCR2 hanya membaca label visual; sistem belum melakukan Optical Chemical Structure Recognition tervalidasi menjadi SMILES/InChI.")
    deduped_limitations: list[str] = []
    for item in limitations:
        if item and item not in deduped_limitations:
            deduped_limitations.append(item)
    return AttachmentContextPackage(
        user_question=user_question,
        file_summary=f"{analysis.filename} diproses via {analysis.extraction_method}.",
        ocr_text=analysis.extracted_text,
        detected_type=detected_type,
        extracted_entities=analysis.detected_entities,
        verified_candidates=analysis.neo4j_candidates,
        neo4j_evidence=analysis.neo4j_candidates,
        confidence=analysis.confidence,
        verification_status=analysis.verification_status,
        limitations=deduped_limitations,
    )


def format_attachment_context_package(package: AttachmentContextPackage) -> str:
    entities = "\n".join(
        f"- {entity.get('type')}: {entity.get('value')}" for entity in package.extracted_entities[:30]
    ) or "Tidak ada entitas eksplisit yang terdeteksi."
    candidates = "\n".join(
        f"- {candidate.get('name', 'unknown')} | score={float(candidate.get('score') or 0):.2f} | evidence={', '.join(candidate.get('matched_evidence') or [])}"
        for candidate in package.verified_candidates[:8]
    ) or "Tidak ada kandidat Neo4j yang cukup kuat."
    limitations = "\n".join(f"- {item}" for item in package.limitations[:12]) or "Tidak ada."
    return f"""[ATTACHMENT EVIDENCE]
Filename:
{package.file_summary}

Detected content type:
{package.detected_type}

OCR result:
{package.ocr_text[:settings.ATTACHMENT_CONTEXT_MAX_CHARS]}

Entities extracted:
{entities}

Neo4j candidates:
{candidates}

Verification status:
{package.verification_status}

Confidence:
{package.confidence:.2f}

Limitations:
{limitations}
[/ATTACHMENT EVIDENCE]"""
