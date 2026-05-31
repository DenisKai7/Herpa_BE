"""
LLM Formatter - Generates AI responses using HuggingFace Inference API.

Fitur:
- Dynamic persona-based system prompts untuk personalized responses.
- STRICT ZERO-HALLUCINATION: AI hanya menjawab berdasarkan database context.
- Mendukung mode blocking (full response) dan streaming (SSE token-by-token).
- File context injection untuk multimodal support (OCR result).
- Menggunakan HuggingFace Inference API via OpenAI-compatible endpoint.

Temperature:
- 0.0 untuk standard response (deterministic, zero-hallucination).
- 0.2 untuk quiz generation (sedikit variasi terkontrol).
"""

import logging
from typing import Generator, Optional

from huggingface_hub import InferenceClient

from app.core.config import settings

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════
# LLM CLIENT (Singleton) - HuggingFace Inference API
# ═══════════════════════════════════════════
_client = InferenceClient(
    provider="auto",
    api_key=settings.HF_API_TOKEN,
)


# ═══════════════════════════════════════════
# PERSONA DEFINITIONS
# Setiap persona memiliki gaya bahasa dan kedalaman yang berbeda
# ═══════════════════════════════════════════
PERSONA_PROMPTS: dict[str, dict[str, str]] = {
    "Tenaga Medis": {
        "style": "klinis, profesional, dan evidence-based",
        "depth": (
            "Gunakan terminologi medis/farmasi. Sertakan nama latin, dosis, "
            "mekanisme aksi, kontraindikasi, dan interaksi obat jika tersedia "
            "dalam data. Tulis dalam format laporan klinis yang ringkas."
        ),
        "greeting": "Rekan sejawat",
    },
    "Peneliti": {
        "style": "akademis, analitis, dan metodologis",
        "depth": (
            "Sertakan referensi metodologi (IC50, LD50, GC-MS, HPLC). "
            "Jelaskan struktur kimia, jalur biosintesis, dan data farmakologis. "
            "Gunakan bahasa ilmiah formal. Soroti gap penelitian jika ada."
        ),
        "greeting": "Peneliti",
    },
    "Pelajar": {
        "style": "edukatif, mudah dipahami, dan bertahap",
        "depth": (
            "Jelaskan konsep dari dasar. Gunakan analogi sederhana. "
            "Berikan poin-poin ringkas dan contoh konkret. Tambahkan "
            "'Poin Penting' dan 'Istilah Kunci' di akhir jika relevan."
        ),
        "greeting": "Pelajar",
    },
    "Umum": {
        "style": "ramah, informatif, dan praktis",
        "depth": (
            "Gunakan bahasa sehari-hari yang mudah dipahami. Fokus pada "
            "manfaat praktis dan cara penggunaan. Hindari jargon teknis, "
            "tetapi tetap akurat secara ilmiah."
        ),
        "greeting": "Pengguna",
    },
}

# ═══════════════════════════════════════════
# INTENT-SPECIFIC INSTRUCTIONS
# ═══════════════════════════════════════════
INTENT_INSTRUCTIONS: dict[str, str] = {
    "konsultasi": (
        "Berikan rekomendasi tanaman obat/herbal berdasarkan gejala. "
        "Sertakan cara penggunaan, dosis umum, dan peringatan. "
        "SELALU ingatkan untuk berkonsultasi dengan tenaga medis profesional."
    ),
    "ensiklopedia": (
        "Berikan informasi ensiklopedis yang lengkap: deskripsi, klasifikasi, "
        "habitat, kandungan fitokimia, dan khasiat. Format sebagai entri "
        "referensi yang terstruktur."
    ),
    "edukasi": (
        "Jelaskan materi edukasi secara bertahap dan terstruktur. "
        "Gunakan format pembelajaran: Definisi -> Penjelasan -> "
        "Contoh -> Kesimpulan."
    ),
}


def _build_system_prompt(
    query: str,
    context: str,
    ai_mode: str,
    intent: str,
    file_context: Optional[str] = None,
) -> str:
    """
    Membangun system prompt lengkap dengan persona, instruksi, dan konteks.

    Args:
        query: Query asli pengguna (untuk logging context, tidak dimasukkan ke prompt).
        context: Konteks dari GraphRAG retriever (vector + graph results).
        ai_mode: Persona AI yang dipilih user.
        intent: Intent yang terdeteksi oleh NLU router.
        file_context: Teks dari file upload OCR (opsional).

    Returns:
        System prompt string yang siap dikirim ke LLM.
    """
    persona = PERSONA_PROMPTS.get(ai_mode, PERSONA_PROMPTS["Umum"])
    intent_instruction = INTENT_INSTRUCTIONS.get(
        intent, "Jawab pertanyaan berdasarkan data yang tersedia."
    )

    file_instruction = ""
    if file_context:
        file_instruction = f"""
═══ ISI FILE UPLOAD PENGGUNA ═══
{file_context[:3000]}
═══ AKHIR FILE UPLOAD ═══
Gunakan isi file di atas sebagai konteks tambahan untuk menjawab pertanyaan."""

    return f"""Anda adalah Asisten AI Farmasi & Edukasi untuk Ensiklopedia Tanaman Obat Indonesia.

═══ IDENTITAS PERSONA ═══
Target pengguna: {persona['greeting']} ({ai_mode})
Gaya bahasa: {persona['style']}
Kedalaman: {persona['depth']}

═══ INSTRUKSI MUTLAK (ZERO-HALLUCINATION) ═══
1. HANYA gunakan informasi dari [DATA DATABASE] yang disediakan di bawah.
2. DILARANG KERAS menggunakan pengetahuan bawaan/internal Anda.
3. Jika data tidak tersedia, jawab: "Maaf, informasi ini belum tersedia dalam database kami."
4. JANGAN mengarang data, angka, atau referensi yang tidak ada dalam konteks.
5. Jika data parsial tersedia, jawab sejauh data yang ada dan nyatakan keterbatasannya.

═══ INSTRUKSI INTENT: {intent.upper()} ═══
{intent_instruction}

═══ FORMAT JAWABAN ═══
- Gunakan markdown untuk formatting.
- Struktur jawaban dengan heading, bullet points, dan penekanan yang tepat.
- Akhiri dengan disclaimer medis jika intent adalah 'konsultasi'.
{file_instruction}
═══ DATA DATABASE MULAI ═══
{context}
═══ DATA DATABASE SELESAI ═══"""


def generate_strict_response(
    query: str,
    context: str,
    ai_mode: str,
    intent: str,
    file_context: Optional[str] = None,
) -> str:
    """
    Generate respons AI dengan STRICT ZERO-HALLUCINATION (blocking mode).

    Flow:
    1. Pilih persona berdasarkan ai_mode (Tenaga Medis/Peneliti/Pelajar/Umum).
    2. Pilih instruksi berdasarkan intent (konsultasi/ensiklopedia/edukasi).
    3. Sisipkan database context ke system prompt.
    4. Jika ada file_context dari OCR, sisipkan juga.
    5. Kirim ke LLM dengan temperature=0.0 (deterministic).

    Args:
        query: Pesan teks dari pengguna.
        context: Konteks dari GraphRAG retriever.
        ai_mode: Persona AI (Tenaga Medis/Peneliti/Pelajar/Umum).
        intent: Intent yang terdeteksi (konsultasi/ensiklopedia/edukasi).
        file_context: Teks dari file upload OCR (opsional).

    Returns:
        String respons AI yang terformat.
    """
    system_prompt = _build_system_prompt(query, context, ai_mode, intent, file_context)

    try:
        res = _client.chat.completions.create(
            model=settings.LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ],
            temperature=0.0,
            max_tokens=2048,
        )

        content = res.choices[0].message.content
        if content is None:
            logger.warning("LLM returned None content, returning fallback message.")
            return "Maaf, tidak ada respons yang dihasilkan. Silakan coba lagi."

        return content

    except Exception as e:
        logger.error(f"HuggingFace LLM generation error: {e}", exc_info=True)
        return (
            f"Maaf, terjadi kesalahan saat memproses permintaan Anda. "
            f"Silakan coba lagi nanti. (Error: {type(e).__name__})"
        )


def generate_streaming_response(
    query: str,
    context: str,
    ai_mode: str,
    intent: str,
    file_context: Optional[str] = None,
) -> Generator[str, None, None]:
    """
    Generator yang menghasilkan streaming token dari LLM.

    Digunakan untuk Server-Sent Events (SSE) di endpoint chat.
    Setiap yield menghasilkan satu chunk teks dari response LLM.

    Args:
        query: Pesan teks dari pengguna.
        context: Konteks dari GraphRAG retriever.
        ai_mode: Persona AI.
        intent: Intent yang terdeteksi.
        file_context: Teks dari file upload OCR (opsional).

    Yields:
        String token/chunk dari LLM response.
    """
    system_prompt = _build_system_prompt(query, context, ai_mode, intent, file_context)

    try:
        stream = _client.chat.completions.create(
            model=settings.LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ],
            temperature=0.0,
            max_tokens=2048,
            stream=True,
        )
        for chunk in stream:
            delta_content = chunk.choices[0].delta.content
            if delta_content:
                yield delta_content

    except Exception as e:
        logger.error(f"HuggingFace LLM streaming error: {e}", exc_info=True)
        yield f"[Error: {type(e).__name__}]"
