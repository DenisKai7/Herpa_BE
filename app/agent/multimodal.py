"""
Multimodal File Processor - Ekstraksi teks dari file upload (PDF, Image, TXT).

Modul ini bertanggung jawab untuk:
- Membaca file PDF menggunakan PyMuPDF (fitz).
- Melakukan OCR pada gambar menggunakan Tesseract (ind+eng).
- Membaca file teks biasa (TXT/CSV).

Hasil ekstraksi digunakan sebagai file_context dalam pipeline AI.
"""

import base64
import io
import logging
import os
from typing import Final

import fitz  # PyMuPDF
import pytesseract
from huggingface_hub import InferenceClient
from PIL import Image

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════
# VISION LLM CLIENT (Fallback untuk OCR kosong)
# ═══════════════════════════════════════════
_vision_client = InferenceClient(
    provider="auto",
    api_key=os.getenv("HF_API_TOKEN", ""),
)

# Model multimodal/vision untuk deskripsi gambar
_VISION_MODEL = "meta-llama/Llama-3.2-11B-Vision-Instruct"

# Prompt analitis untuk vision model
_VISION_ANALYSIS_PROMPT = (
    "Analyze this medical/chemical image closely. "
    "Extract and write a comprehensive, factual textual description "
    "of all molecular formulas, chemical compound graphs, medicinal plant "
    "structures, diagnostic lab charts, or any text visible in this image. "
    "Write the description in Indonesian (Bahasa Indonesia). "
    "If you see a plant, describe its morphology, leaf shape, flower, "
    "and any identifiable botanical features. "
    "If you see chemical structures, describe the functional groups, "
    "bonds, and molecular formula. "
    "Be as detailed and factual as possible."
)

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
    Mengekstrak teks dari gambar menggunakan Tesseract OCR.
    Jika OCR menghasilkan < 10 karakter, otomatis fallback ke
    Vision LLM untuk deskripsi visual analitis.

    Args:
        file_bytes: Bytes konten file gambar.

    Returns:
        Teks yang diekstrak dari gambar via OCR atau deskripsi Vision LLM.

    Raises:
        RuntimeError: Jika OCR dan Vision fallback gagal.
    """
    try:
        image = Image.open(io.BytesIO(file_bytes))
        extracted_text: str = pytesseract.image_to_string(image, lang="ind+eng")

        ocr_clean = extracted_text.strip()
        logger.info(f"OCR extracted: {len(ocr_clean)} chars from image.")

        # Jika OCR berhasil mengekstrak teks yang cukup, gunakan hasilnya
        if len(ocr_clean) >= 10:
            return extracted_text

        # ── FALLBACK: Vision LLM Description ──
        logger.warning(
            f"OCR yielded only {len(ocr_clean)} chars (< 10). "
            "Triggering Vision LLM fallback for image description."
        )
        vision_description = _describe_image_with_vision(file_bytes)
        if vision_description and vision_description.strip():
            return vision_description

        # Jika Vision juga gagal, return apapun yang OCR hasilkan
        logger.warning("Vision LLM fallback also produced no output. Using raw OCR result.")
        return extracted_text

    except Exception as e:
        logger.error(f"Image extraction failed: {e}", exc_info=True)
        raise RuntimeError(f"Gagal memproses gambar: {e}") from e


def _describe_image_with_vision(file_bytes: bytes) -> str:
    """
    Mendeskripsikan konten visual gambar menggunakan Vision LLM (VLM).

    Digunakan sebagai fallback ketika OCR menghasilkan teks kosong/minimal,
    misalnya untuk gambar struktur kimia, diagram, atau foto tanaman obat
    yang tidak mengandung teks tercetak.

    Args:
        file_bytes: Bytes konten file gambar.

    Returns:
        String deskripsi visual dalam Bahasa Indonesia.
        Mengembalikan string kosong jika Vision LLM gagal.
    """
    api_key = os.getenv("HF_API_TOKEN", "")
    if not api_key:
        logger.warning("HF_API_TOKEN not set, cannot call Vision LLM.")
        return ""

    try:
        # Encode gambar ke base64 data URI
        b64_image = base64.b64encode(file_bytes).decode("utf-8")

        # Deteksi MIME type dari header bytes
        mime = "image/jpeg"
        if file_bytes[:8].startswith(b"\x89PNG"):
            mime = "image/png"
        elif file_bytes[:4].startswith(b"RIFF") and file_bytes[8:12] == b"WEBP":
            mime = "image/webp"

        data_uri = f"data:{mime};base64,{b64_image}"

        logger.info(
            f"Calling Vision LLM ({_VISION_MODEL}) for image description "
            f"({len(file_bytes)} bytes, {mime})..."
        )

        response = _vision_client.chat.completions.create(
            model=_VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": data_uri},
                        },
                        {
                            "type": "text",
                            "text": _VISION_ANALYSIS_PROMPT,
                        },
                    ],
                }
            ],
            max_tokens=1024,
        )

        description = response.choices[0].message.content or ""
        logger.info(
            f"Vision LLM returned {len(description)} chars of image description."
        )
        return description.strip()

    except Exception as e:
        logger.error(
            f"Vision LLM description failed: {e}. "
            "Falling back to empty OCR result.",
            exc_info=True,
        )
        return ""


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
