"""
Multimodal File Processor - Ekstraksi teks dari file upload (PDF, Image, TXT).

Modul ini bertanggung jawab untuk:
- Membaca file PDF menggunakan PyMuPDF (fitz).
- Mengonversi gambar menjadi deskripsi kimia/medis analitis menggunakan Groq Vision API.
- Membaca file teks biasa (TXT/CSV).

Hasil ekstraksi digunakan sebagai file_context dalam pipeline AI.
"""

import base64
import io
import logging
import os
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
    Membuat string Base64 yang bersih dari karakter whitespace tersembunyi
    untuk mencegah eror 400 Bad Request pada API Gateway.
    """
    base64_encoded = base64.b64encode(file_bytes).decode("utf-8")
    return base64_encoded.replace("\n", "").replace("\r", "").strip()


def _describe_image_with_vision(file_bytes: bytes) -> dict[str, str]:
    """
    Mendeskripsikan konten visual gambar menggunakan Groq Cloud API
    dengan model llama-3.2-11b-vision-instruct.
    """
    # Deteksi MIME type dari header bytes
    mime_type = "image/jpeg"
    if file_bytes[:8].startswith(b"\x89PNG"):
        mime_type = "image/png"
    elif file_bytes[:4].startswith(b"RIFF") and file_bytes[8:12] == b"WEBP":
        mime_type = "image/webp"

    try:
        # Enkripsi ke Base64 string yang steril tanpa newline characters
        base64_image_string = encode_image_to_base64(file_bytes)

        # Instruksi prompt fitokimia ketat dengan pembagian struktural molekul
        prompt_instruction = (
            "Bertindaklah sebagai Ahli Spektroskopi dan Laboratorium Fitokimia. Analisis gambar struktur kimia ini dengan presisi tinggi.\n"
            "Perhatikan ciri utama ini untuk mencegah salah tebak:\n"
            "1. Cek Cincin Aromatik Benzena: Jika memiliki DUA cincin benzena simetris di ujung kiri dan kanan dengan gugus -OH dan -OCH3, ini adalah KURKUMIN / KURKUMINOID (Kunyit).\n"
            "2. Cek Cincin Tunggal Fenolik: Jika hanya memiliki SATU cincin benzena fenolik (gugus -OH) yang terikat pada rantai hidrokarbon tidak jenuh seskuiterpenoid (rantai karbon dengan ikatan rangkap dua alkena), ini adalah XANTHORRHIZOL (Temulawak).\n"
            "3. Cek Asimetris Gugus Alkil: Jika hanya memiliki SATU cincin aromatik benzena dengan rantai hidrokarbon lurus panjang yang mengandung gugus hidroksil dan keton jenuh, ini adalah GINGEROL (Jahe).\n\n"
            "Tulis deskripsi ringkas dalam Bahasa Indonesia yang edukatif untuk mahasiswa farmasi/informatika.\n\n"
            "CRITICAL REQUIREMENT: Di baris paling akhir dari jawaban Anda, Anda WAJIB menuliskan satu baris penutup dengan format kaku:\n"
            "[TARGET: Nama_Senyawa_Murni]"
        )

        payload = {
            "model": "llama-3.2-11b-vision-instruct",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt_instruction
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
            "temperature": 0.0,
            "max_tokens": 800
        }

        headers = {
            "Authorization": f"Bearer {settings.GROQ_API_TOKEN or settings.HF_API_TOKEN}",
            "Content-Type": "application/json"
        }

        url = "https://api.groq.com/openai/v1/chat/completions"

        logger.info("Calling Groq llama-3.2-11b-vision-instruct with sanitized base64 sequence...")
        response = httpx.post(url, json=payload, headers=headers, timeout=60.0)
        
        # Jika server menolak, tangkap isi teks penolakan aslinya untuk log terminal
        if response.status_code != 200:
            logger.error(f"Groq Cloud gateway rejected request. Error payload: {response.text}")
            response.raise_for_status()

        response_json = response.json()
        generated_text = response_json["choices"][0]["message"]["content"]
        return {"extracted_text_content": generated_text}

    except Exception as e:
        logger.warning(f"Vision LLM description failed: {e}", exc_info=True)
        error_detail = f" (Details: {str(e)})"
        if 'response' in locals() and hasattr(response, 'text'):
            error_detail += f" | Server Response: {response.text}"

        # Proteksi Fallback Bersih: Gagalkan tag target agar pencarian grafik tahu objek tidak terbaca
        return {
            "extracted_text_content": (
                f"[TARGET: Unknown] Terdeteksi lampiran gambar berkas struktur kimia, "
                f"namun sistem pemrosesan visual sedang sibuk.{error_detail}"
            )
        }


def _extract_from_pdf(file_bytes: bytes) -> str:
    """
    Mengekstrak teks dari file PDF menggunakan PyMuPDF.
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
    """
    result = _describe_image_with_vision(file_bytes)
    return result["extracted_text_content"]


def _extract_from_text(file_bytes: bytes) -> str:
    """
    Membaca konten file teks biasa.
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