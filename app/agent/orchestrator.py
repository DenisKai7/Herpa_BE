"""
Orchestrator - Central Agentic Pipeline.

Mengorkestrasi seluruh flow permintaan chat:
1. Multimodal Interception: Jika ada file_context, sisipkan ke prompt.
2. NLU Intent Routing: Klasifikasi query via SVM model.
3. GraphRAG Retrieval: Hybrid search (vector + graph).
4. Agentic Execution: LLM formatting atau Quiz Tool-Calling Agent.

Mendukung dua mode:
- Blocking (process_user_query): Response lengkap sekaligus.
- Streaming (process_user_query_stream): SSE token-by-token.

Semua operasi I/O berat (retrieval, LLM call) dijalankan melalui
asyncio thread pool agar tidak memblokir event loop FastAPI.
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, AsyncGenerator, Generator, Optional

from app.agent.router import classify_intent
from app.agent.retriever import (
    content_based_recommendation,
    search_encyclopedia,
    retrieve_education_corpus,
)
from app.agent.llm_formatter import generate_strict_response, generate_streaming_response
from app.agent.quiz_generator import generate_interactive_quiz_tool

logger = logging.getLogger(__name__)

# Thread pool untuk blocking I/O (model inference, DB queries, LLM calls)
_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="agent-worker")


# ═══════════════════════════════════════════
# INTERNAL HELPERS
# ═══════════════════════════════════════════

def _preprocess_file_context(file_context: Optional[str]) -> Optional[str]:
    """
    Defensive preprocessing untuk file context dari upload.

    Melakukan:
    1. Validasi bahwa string tidak empty.
    2. Truncate ke 3000 karakter untuk mencegah context window overflow.
    3. Strip whitespace berlebih.

    Args:
        file_context: Raw file context text dari OCR/ekstraksi.

    Returns:
        Cleaned dan truncated file context, atau None jika tidak ada.
    """
    if not file_context or not file_context.strip():
        return None

    # Strip whitespace
    cleaned = file_context.strip()

    # Defensive truncation — 3000 chars to leave room for directive text
    # that gets appended in llm_formatter (prevents context window overflow)
    file_context_buffer = cleaned
    if len(file_context_buffer) > 3000:
        logger.warning(
            f"File context too long ({len(file_context_buffer)} chars), "
            f"truncating to 3000 chars to prevent LLM context overflow."
        )
        file_context_buffer = file_context_buffer[:3000]
    cleaned = file_context_buffer

    logger.info(
        f"File context preprocessed: {len(cleaned)} chars "
        f"(original: {len(file_context)} chars)"
    )
    return cleaned


def _retrieve_context(intent: str, query: str) -> str:
    """
    Memilih dan menjalankan retriever berdasarkan intent yang terdeteksi.

    Args:
        intent: Intent hasil klasifikasi NLU.
        query: Query asli dari pengguna.

    Returns:
        String konteks gabungan dari vector search + graph search.
    """
    retriever_map: dict[str, Any] = {
        "konsultasi": content_based_recommendation,
        "ensiklopedia": search_encyclopedia,
        "edukasi": retrieve_education_corpus,
    }

    retriever = retriever_map.get(intent)
    if retriever:
        return retriever(query)

    logger.warning(f"No retriever found for intent '{intent}', returning domain notice.")
    return "Sistem hanya melayani domain farmasi, herbal, tanaman obat, dan kimia terkait."


# ═══════════════════════════════════════════
# BLOCKING PIPELINE (Non-Streaming)
# ═══════════════════════════════════════════

def _process_query_sync(
    query: str,
    ai_mode: str,
    file_context: Optional[str] = None,
    model: Optional[str] = None,
) -> dict[str, Any]:
    """
    Pipeline sinkron untuk memproses query user (dijalankan di thread pool).

    Flow:
    1. Classify intent via NLP Router (SVM).
    2. Jika 'generate_quiz' -> panggil Quiz Tool-Calling Agent.
    3. Selain itu -> RAG retrieval -> LLM formatting.

    Args:
        query: Pesan teks dari pengguna.
        ai_mode: Persona AI (Tenaga Medis/Peneliti/Pelajar/Umum).
        file_context: Teks hasil OCR dari file upload (opsional).
        model: Model LLM yang sudah tervalidasi berdasarkan role (opsional).

    Returns:
        Dict berisi intent_detected, ai_response, dan opsional quiz_payload.
    """
    intent = classify_intent(query)
    logger.info(
        f"Intent classified: '{intent}' | "
        f"Query: '{query[:80]}...' | Mode: {ai_mode}"
    )

    # ── CAPTURE UPLOADED FILE PAYLOAD DATA ──
    # Validate, strip whitespace, and truncate to 3000 chars to prevent
    # 422 Unprocessable Entity crashes from context window overflow.
    file_data_content = _preprocess_file_context(file_context)
    if file_data_content:
        logger.info(
            f"File context captured: {len(file_data_content)} chars "
            f"(will be injected into LLM system prompt)"
        )

    # ── QUIZ INTENT: Agentic Tool-Calling ──
    if intent == "generate_quiz":
        try:
            quiz_data = generate_interactive_quiz_tool(
                topic=query,
                jumlah_soal=3,
                ai_mode=ai_mode,
                file_context=file_data_content,
                model=model,
            )
            return {
                "intent_detected": "quiz_rendered",
                "quiz_payload": quiz_data,
                "ai_response": (
                    "Berikut adalah kuis interaktif Anda "
                    "berdasarkan materi yang tersedia."
                ),
            }
        except Exception as e:
            logger.error(f"Quiz generation failed: {e}", exc_info=True)
            return {
                "intent_detected": "generate_quiz",
                "ai_response": (
                    f"Maaf, gagal membuat kuis. "
                    f"Silakan coba lagi. ({type(e).__name__})"
                ),
            }

    # ── RAG INTENT: Retrieval + LLM Formatting ──
    search_query = query
    if file_data_content:
        # Combine user question with the factual vision analysis text from Groq
        search_query = f"{query} {file_data_content.strip()}"

    context_data = _retrieve_context(intent, search_query)

    # Strict RAG Guardrail Bypass Layer
    if (not context_data or context_data.strip() == "" or "0 records found" in context_data) and file_data_content:
        logger.info("Graph search was empty but file context exists. Feeding vision data into context_data to bypass strict RAG constraints.")
        context_data = f"Data Hasil Analisis Gambar/Berkas Laboratorium: {file_data_content}"

    final_response = generate_strict_response(
        query=query,
        context=context_data,
        ai_mode=ai_mode,
        intent=intent,
        file_context=file_data_content,
        model=model,
    )

    return {
        "intent_detected": intent,
        "ai_response": final_response,
    }


async def process_user_query(
    query: str,
    ai_mode: str,
    file_context: Optional[str] = None,
    model: Optional[str] = None,
) -> dict[str, Any]:
    """
    Pipeline utama (non-streaming, async) untuk memproses query user.

    Menjalankan pipeline sinkron di thread pool agar tidak memblokir
    event loop FastAPI.

    Args:
        query: Pesan teks dari pengguna.
        ai_mode: Persona AI (Tenaga Medis/Peneliti/Pelajar/Umum).
        file_context: Teks hasil OCR dari file upload (opsional).
        model: Model LLM yang sudah tervalidasi berdasarkan role (opsional).

    Returns:
        Dict berisi intent_detected, ai_response, dan opsional quiz_payload.
    """
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        _executor,
        _process_query_sync,
        query,
        ai_mode,
        file_context,
        model,
    )
    return result


# ═══════════════════════════════════════════
# STREAMING PIPELINE (SSE)
# ═══════════════════════════════════════════

def _stream_query_sync(
    query: str,
    ai_mode: str,
    file_context: Optional[str] = None,
    model: Optional[str] = None,
) -> Generator[dict[str, Any], None, None]:
    """
    Pipeline sinkron streaming untuk SSE (dijalankan di thread pool).

    Menghasilkan event dict secara berurutan:
    1. Event 'intent' dengan intent yang terdeteksi.
    2. Event 'token' untuk setiap chunk LLM response (atau 'quiz' untuk quiz).
    3. Event 'full_response' dengan response lengkap untuk DB storage.
    4. Event 'done' saat selesai.

    Args:
        query: Pesan teks dari pengguna.
        ai_mode: Persona AI.
        file_context: Teks hasil OCR (opsional).
        model: Model LLM yang sudah tervalidasi berdasarkan role (opsional).

    Yields:
        Dict dengan keys 'event' dan 'data'.
    """
    intent = classify_intent(query)
    logger.info(f"[Stream] Intent: '{intent}' | Query: '{query[:80]}...'")

    # Yield intent terlebih dahulu
    yield {"event": "intent", "data": intent}

    # ── CAPTURE UPLOADED FILE PAYLOAD DATA ──
    file_data_content = _preprocess_file_context(file_context)
    if file_data_content:
        logger.info(
            f"[Stream] File context captured: {len(file_data_content)} chars "
            f"(will be injected into streaming LLM prompt)"
        )

    # ── QUIZ: Return full payload (tidak di-stream) ──
    if intent == "generate_quiz":
        try:
            quiz_data = generate_interactive_quiz_tool(
                topic=query,
                jumlah_soal=3,
                ai_mode=ai_mode,
                file_context=file_data_content,
                model=model,
            )
            yield {
                "event": "quiz",
                "data": {
                    "quiz_payload": quiz_data,
                    "message": (
                        "Berikut adalah kuis interaktif Anda "
                        "berdasarkan materi yang tersedia."
                    ),
                },
            }
        except Exception as e:
            logger.error(f"Quiz generation failed during stream: {e}", exc_info=True)
            yield {"event": "error", "data": str(e)}
        yield {"event": "done", "data": ""}
        return

    # ── RAG + Streaming LLM ──
    search_query = query
    if file_data_content:
        search_query = f"{query} {file_data_content.strip()}"

    context_data = _retrieve_context(intent, search_query)

    # Strict RAG Guardrail Bypass Layer for Stream
    if (not context_data or context_data.strip() == "" or "0 records found" in context_data) and file_data_content:
        logger.info("[Stream] Graph search empty, bypassing strict RAG constraint via vision data injection.")
        context_data = f"Data Hasil Analisis Gambar/Berkas Laboratorium: {file_data_content}"

    full_response = ""
    try:
        for token in generate_streaming_response(
            query=query,
            context=context_data,
            ai_mode=ai_mode,
            intent=intent,
            file_context=file_data_content,
            model=model,
        ):
            full_response += token
            yield {"event": "token", "data": token}
    except Exception as e:
        logger.error(f"Streaming LLM error: {e}", exc_info=True)
        yield {"event": "error", "data": str(e)}

    # Yield full response untuk penyimpanan di DB
    yield {"event": "full_response", "data": full_response}
    yield {"event": "done", "data": ""}


async def process_user_query_stream(
    query: str,
    ai_mode: str,
    file_context: Optional[str] = None,
    model: Optional[str] = None,
) -> AsyncGenerator[dict[str, Any], None]:
    """
    Pipeline streaming async untuk SSE endpoint.

    Menjalankan generator sinkron di thread pool dan mengkonversi
    ke async generator agar kompatibel dengan FastAPI StreamingResponse.

    Args:
        query: Pesan teks dari pengguna.
        ai_mode: Persona AI.
        file_context: Teks hasil OCR (opsional).
        model: Model LLM yang sudah tervalidasi berdasarkan role (opsional).

    Yields:
        Dict event untuk SSE (intent, token, quiz, full_response, done, error).
    """
    loop = asyncio.get_event_loop()
    import queue

    event_queue: queue.Queue[dict[str, Any] | None] = queue.Queue()

    def _run_stream() -> None:
        """Worker: jalankan stream sync dan masukkan event ke queue."""
        try:
            for event in _stream_query_sync(query, ai_mode, file_context, model):
                event_queue.put(event)
        except Exception as e:
            logger.error(f"Stream worker error: {e}", exc_info=True)
            event_queue.put({"event": "error", "data": str(e)})
        finally:
            event_queue.put(None)  # Sentinel: stream selesai

    # Jalankan sync generator di thread pool
    loop.run_in_executor(_executor, _run_stream)

    # Consume events dari queue secara async
    while True:
        try:
            event = await loop.run_in_executor(None, event_queue.get, True, 120.0)
        except Exception:
            logger.error("Stream event queue timeout (120s).")
            yield {"event": "error", "data": "Stream timeout."}
            break

        if event is None:
            break

        yield event
