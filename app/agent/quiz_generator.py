"""
Quiz Generator - Agentic Tool-Calling untuk pembuatan kuis interaktif.

Menggunakan HuggingFace Inference API Tool-Calling untuk memaksa output
dalam format JSON ketat sesuai QuizResponse schema (Pydantic).
Temperature 0.2 untuk variasi soal yang terkontrol.

Pipeline:
1. Retrieve konteks edukasi dari GraphRAG retriever.
2. Bangun system prompt dengan konteks database + file upload.
3. Definisikan tool schema dari QuizResponse Pydantic model.
4. Panggil LLM dengan tool_choice forced ke render_interactive_quiz.
5. Parse dan validasi response terhadap Pydantic schema.
"""

import json
import logging
from typing import Any, Optional

from huggingface_hub import InferenceClient

from app.core.config import settings
from app.agent.retriever import retrieve_education_corpus
from app.models.quiz_schemas import QuizResponse

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════
# LLM CLIENT (Shared, Singleton) - HuggingFace Inference API
# ═══════════════════════════════════════════
_client = InferenceClient(
    provider="auto",
    api_key=settings.HF_API_TOKEN,
)


def _build_quiz_system_prompt(
    context_data: str,
    jumlah_soal: int,
    ai_mode: str,
    file_context: Optional[str] = None,
) -> str:
    """
    Membangun system prompt untuk quiz generation.

    Args:
        context_data: Konteks dari GraphRAG retriever.
        jumlah_soal: Jumlah soal yang diminta.
        ai_mode: Persona AI (Pelajar, Tenaga Medis, dll).
        file_context: Teks dari file upload (opsional).

    Returns:
        System prompt string yang siap dikirim ke LLM.
    """
    file_instruction = ""
    if file_context:
        truncated = file_context[:2000]
        file_instruction = (
            f"\nFokuskan juga soal dari teks referensi file "
            f"yang diunggah pengguna berikut:\n{truncated}\n"
        )

    return f"""Anda adalah Sistem Pembuat Kuis Farmasi & Kimia yang ketat dan akurat.
Target pengguna: {ai_mode}.

═══ INSTRUKSI MUTLAK ═══
1. Buat TEPAT {jumlah_soal} soal berdasarkan [DATA DATABASE] di bawah.
2. HANYA gunakan informasi dari data yang disediakan.
3. JANGAN mengarang informasi yang tidak ada dalam data.
4. Setiap soal HARUS memiliki tepat 4 opsi jawaban (A, B, C, D).
5. Pembahasan harus merujuk pada data database, bukan pengetahuan umum.
6. Variasikan tingkat kesulitan: Mudah, Menengah, dan HOTS.
{file_instruction}
═══ DATA DATABASE MULAI ═══
{context_data}
═══ DATA DATABASE SELESAI ═══"""


def _build_tool_schema() -> list[dict[str, Any]]:
    """
    Membangun definisi tool untuk OpenAI Tool-Calling.

    Returns:
        List berisi satu tool definition berbasis QuizResponse schema.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "render_interactive_quiz",
                "description": (
                    "Merender kuis interaktif dengan soal pilihan ganda, "
                    "jawaban benar, dan pembahasan langkah demi langkah."
                ),
                "parameters": QuizResponse.model_json_schema(),
            },
        }
    ]


def generate_interactive_quiz_tool(
    topic: str,
    jumlah_soal: int = 3,
    ai_mode: str = "Pelajar",
    file_context: Optional[str] = None,
) -> dict[str, Any]:
    """
    Generate kuis interaktif menggunakan LLM Tool-Calling.

    Pipeline:
    1. Retrieve konteks edukasi dari database (vector + graph).
    2. Kirim ke LLM dengan tool schema QuizResponse.
    3. Parse tool call arguments dari response.
    4. Validasi terhadap Pydantic schema.

    Args:
        topic: Topik kuis yang diminta pengguna.
        jumlah_soal: Jumlah soal (default: 3).
        ai_mode: Persona AI target (default: Pelajar).
        file_context: Teks dari file upload pengguna (opsional).

    Returns:
        Dict berisi quiz data sesuai QuizResponse schema.

    Raises:
        ValueError: Jika response LLM tidak mengandung tool call yang valid.
        RuntimeError: Jika LLM API call gagal.
    """
    logger.info(
        f"Generating quiz: topic='{topic[:50]}', "
        f"jumlah_soal={jumlah_soal}, mode={ai_mode}"
    )

    # Step 1: Retrieve education context
    context_data = retrieve_education_corpus(topic)

    # Step 2: Build prompt dan tool schema
    system_prompt = _build_quiz_system_prompt(
        context_data, jumlah_soal, ai_mode, file_context
    )
    tools = _build_tool_schema()

    # Step 3: Call LLM with forced tool choice
    try:
        response = _client.chat.completions.create(
            model=settings.LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Buat kuis tentang: {topic}"},
            ],
            tools=tools,
            tool_choice={
                "type": "function",
                "function": {"name": "render_interactive_quiz"},
            },
            temperature=0.2,
            max_tokens=4096,
        )
    except Exception as e:
        logger.error(f"HuggingFace LLM API call failed for quiz generation: {e}", exc_info=True)
        raise RuntimeError(
            f"Gagal menghubungi layanan AI HuggingFace untuk membuat kuis: {type(e).__name__}"
        ) from e

    # Step 4: Extract and validate tool call response
    message = response.choices[0].message

    if not message.tool_calls or len(message.tool_calls) == 0:
        logger.error("LLM response did not contain any tool calls.")
        raise ValueError(
            "AI tidak menghasilkan format kuis yang valid. Silakan coba lagi."
        )

    tool_call = message.tool_calls[0]

    if tool_call.function.name != "render_interactive_quiz":
        logger.error(
            f"Unexpected tool call: '{tool_call.function.name}' "
            f"(expected 'render_interactive_quiz')"
        )
        raise ValueError("AI memanggil fungsi yang salah. Silakan coba lagi.")

    try:
        raw_arguments = json.loads(tool_call.function.arguments)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse tool call arguments as JSON: {e}", exc_info=True)
        raise ValueError(
            "AI menghasilkan format data yang tidak valid. Silakan coba lagi."
        ) from e

    # Step 5: Validate against Pydantic schema
    try:
        validated_quiz = QuizResponse.model_validate(raw_arguments)
        logger.info(
            f"Quiz generated successfully: {len(validated_quiz.daftar_soal)} soal, "
            f"topik='{validated_quiz.topik}'"
        )
        return validated_quiz.model_dump()
    except Exception as e:
        logger.warning(
            f"Pydantic validation failed, returning raw data: {e}",
            exc_info=True,
        )
        # Fallback: return raw data jika validasi gagal tapi JSON valid
        return raw_arguments
