"""
Multimodal File Processor - Ekstraksi teks dari file upload (PDF, Image, TXT).

Modul ini bertanggung jawab untuk:
- Membaca file PDF menggunakan PyMuPDF (fitz).
- Mengonversi gambar menjadi deskripsi kimia/medis analitis menggunakan Qwen2.5-VL-7B-Instruct.
- Membaca file teks biasa (TXT/CSV).

Hasil ekstraksi digunakan sebagai file_context dalam pipeline AI.
"""

import base64
import io
import logging
from typing import Final, Optional

import fitz  # PyMuPDF
import httpx
from PIL import Image

from app.core.config import settings

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════
# KONSTANTA FILE PROCESSING
# ═══════════════════════════════════════════

# Ekstensi yang diizinkan, dikelompokkan berdasarkan tipe prosesor
PDF_EXTENSIONS: Final[frozenset[str]] = frozenset({"pdf"})
IMAGE_EXTENSIONS: Final[frozenset[str]] = frozenset({"jpg", "jpeg", "png", "webp"})
TEXT_EXTENSIONS: Final[frozenset[str]] = frozenset({"txt"})
ALL_ALLOWED_EXTENSIONS: Final[frozenset[str]] = PDF_EXTENSIONS | IMAGE_EXTENSIONS | TEXT_EXTENSIONS

# MIME type mapping yang diizinkan (SECURITY: validasi ketat per brief)
ALLOWED_MIME_TYPES: Final[dict[str, frozenset[str]]] = {
    "application/pdf": PDF_EXTENSIONS,
    "image/jpeg": frozenset({"jpg", "jpeg"}),
    "image/png": frozenset({"png"}),
    "image/webp": frozenset({"webp"}),
    "text/plain": TEXT_EXTENSIONS,
}

# Batas maksimal karakter hasil ekstraksi untuk dikirim ke LLM
MAX_EXTRACTED_CHARS: Final[int] = 15_000


def validate_file_type(filename: str, content_type: str | None = None) -> str:
    """
    Validasi tipe file berdasarkan ekstensi dan opsional MIME type.

    Args:
        filename: Nama file asli dari upload.
        content_type: MIME type dari HTTP upload header (opsional).

    Returns:
        Ekstensi file yang tervalidasi (lowercase, tanpa titik).

    Raises:
        ValueError: Jika ekstensi atau MIME type tidak diizinkan.
    """
    if "." not in filename:
        raise ValueError(f"File '{filename}' tidak memiliki ekstensi.")

    ext = filename.rsplit(".", maxsplit=1)[-1].lower()

    if ext not in ALL_ALLOWED_EXTENSIONS:
        raise ValueError(
            f"Format file '.{ext}' tidak didukung. "
            f"Format yang diizinkan: {', '.join(sorted(ALL_ALLOWED_EXTENSIONS))}"
        )

    # Validasi MIME type jika tersedia
    if content_type:
        normalized_mime = content_type.split(";")[0].strip().lower()
        if normalized_mime not in ALLOWED_MIME_TYPES:
            raise ValueError(
                f"MIME type '{normalized_mime}' tidak diizinkan. "
                f"Tipe yang diterima: {', '.join(sorted(ALLOWED_MIME_TYPES.keys()))}"
            )
        # Cross-check: MIME type harus sesuai dengan ekstensi
        allowed_exts_for_mime = ALLOWED_MIME_TYPES[normalized_mime]
        if ext not in allowed_exts_for_mime:
            raise ValueError(
                f"MIME type '{normalized_mime}' tidak cocok dengan ekstensi '.{ext}'."
            )

    return ext


def encode_image_to_base64(file_bytes: bytes) -> str:
    """
    Membaca file gambar dan mengonversi ke base64 encoded string.

    Args:
        file_bytes: Bytes konten file gambar.

    Returns:
        Clean base64 string.
    """
    return base64.b64encode(file_bytes).decode("utf-8")


def _describe_image_with_vision(file_bytes: bytes) -> dict[str, str]:
    """
    Mendeskripsikan konten visual gambar menggunakan Vision LLM Qwen2.5-VL-7B-Instruct.

    Args:
        file_bytes: Bytes konten file gambar.

    Returns:
        Dictionary dengan key 'extracted_text_content' berisi hasil deskripsi.
    """
    # Deteksi MIME type dari header bytes
    mime_type = "image/jpeg"
    if file_bytes[:8].startswith(b"\x89PNG"):
        mime_type = "image/png"
    elif file_bytes[:4].startswith(b"RIFF") and file_bytes[8:12] == b"WEBP":
        mime_type = "image/webp"

    try:
        base64_image_string = encode_image_to_base64(file_bytes)

        payload = {
            "model": "Qwen/Qwen2.5-VL-7B-Instruct",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Bertindaklah sebagai Ahli Laboratorium Farmasi dan Fitokimia. Analisis gambar senyawa aktif/tanaman herbal ini dengan sangat teliti. Tuliskan deskripsi lengkap berisi: 1. Nama senyawa kimia/nama latin tanaman yang terdeteksi, 2. Gugus fungsi atau rumus struktur yang terlihat, 3. Khasiat utamanya secara ilmiah. Tulis dalam Bahasa Indonesia yang padat dan langsung pada inti data tanpa basa-basi."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{base64_image_string}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 800
        }

        headers = {
            "Authorization": f"Bearer {settings.HF_API_TOKEN}",
            "Content-Type": "application/json"
        }

        url = "https://api-inference.huggingface.co/v1/chat/completions"

        logger.info("Calling Qwen2.5-VL-7B-Instruct for image description...")
        response = httpx.post(url, json=payload, headers=headers, timeout=60.0)
        response.raise_for_status()

        response_json = response.json()
        generated_text = response_json["choices"][0]["message"]["content"]
        return {"extracted_text_content": generated_text}

    except Exception as e:
        logger.warning(
            f"Vision LLM description failed: {e}. "
            "Falling back to default message.",
            exc_info=True,
        )
        return {
            "extracted_text_content": (
                "Terdeteksi gambar lampiran struktur kimia/tanaman herbal, "
                "namun Vision API sedang sibuk."
            )
        }


def _extract_from_pdf(file_bytes: bytes) -> str:
    """
    Mengekstrak teks dari file PDF menggunakan PyMuPDF.

    Args:
        file_bytes: Bytes konten file PDF.

    Returns:
        Teks yang diekstrak dari seluruh halaman PDF.

    Raises:
        RuntimeError: Jika PDF tidak bisa dibuka atau corrupt.
    """
    try:
        pdf_document = fitz.open(stream=file_bytes, filetype="pdf")
        pages_text: list[str] = []
        for page_num in range(len(pdf_document)):
            page_text = pdf_document[page_num].get_text("text")
            if page_text.strip():
                pages_text.append(page_text)
        pdf_document.close()

        logger.info(f"PDF extracted: {len(pages_text)} pages with text content.")
        return "\n".join(pages_text)
    except Exception as e:
        logger.error(f"PDF extraction failed: {e}", exc_info=True)
        raise RuntimeError(f"Gagal membaca file PDF: {e}") from e


def _extract_from_image(file_bytes: bytes) -> str:
    """
    Mengekstrak deskripsi visual gambar menggunakan Vision LLM.

    Args:
        file_bytes: Bytes konten file gambar.

    Returns:
        Teks deskripsi visual hasil Vision LLM.
    """
    result = _describe_image_with_vision(file_bytes)
    return result["extracted_text_content"]


def _extract_from_text(file_bytes: bytes) -> str:
    """
    Membaca konten file teks biasa.

    Args:
        file_bytes: Bytes konten file teks (UTF-8).

    Returns:
        String konten file.

    Raises:
        RuntimeError: Jika file tidak bisa di-decode.
    """
    try:
        return file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        try:
            return file_bytes.decode("latin-1")
        except Exception as e:
            logger.error(f"Text file decode failed: {e}", exc_info=True)
            raise RuntimeError(f"Gagal membaca file teks: {e}") from e


def extract_text_from_file(
    file_bytes: bytes,
    filename: str,
    content_type: str | None = None,
) -> str:
    """
    Entry point utama: ekstrak teks dari file berdasarkan tipe.

    Pipeline:
    1. Validasi ekstensi dan MIME type.
    2. Route ke extractor yang sesuai (PDF/Image/Text).
    3. Bersihkan whitespace berlebih.
    4. Truncate jika melebihi batas karakter.

    Args:
        file_bytes: Bytes konten file yang diupload.
        filename: Nama file asli dari user.
        content_type: MIME type dari upload HTTP header.

    Returns:
        Teks yang diekstrak dan dibersihkan, siap dipakai sebagai konteks AI.

    Raises:
        ValueError: Jika tipe file tidak didukung.
        RuntimeError: Jika proses ekstraksi gagal.
    """
    ext = validate_file_type(filename, content_type)

    logger.info(f"Processing file: '{filename}' (ext=.{ext}, mime={content_type})")

    if ext in PDF_EXTENSIONS:
        raw_text = _extract_from_pdf(file_bytes)
    elif ext in IMAGE_EXTENSIONS:
        raw_text = _extract_from_image(file_bytes)
    elif ext in TEXT_EXTENSIONS:
        raw_text = _extract_from_text(file_bytes)
    else:
        # Seharusnya tidak tercapai karena sudah divalidasi di atas
        raise ValueError(f"Format file '.{ext}' tidak didukung oleh sistem.")

    # Bersihkan whitespace berlebih
    cleaned = " ".join(raw_text.split())

    # Truncate jika terlalu panjang untuk konteks LLM
    if len(cleaned) > MAX_EXTRACTED_CHARS:
        logger.warning(
            f"Extracted text truncated from {len(cleaned)} to {MAX_EXTRACTED_CHARS} chars."
        )
        cleaned = cleaned[:MAX_EXTRACTED_CHARS]

    return cleaned
