"""
Orchestrator - Central Agentic Pipeline.
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
from app.agent.llm_formatter import generate_strict_response, generate_streaming_response, MODEL_REGISTRY
from app.agent.quiz_generator import generate_interactive_quiz_tool
from app.core.dependencies import resolve_model, ModelTier

logger = logging.getLogger(__name__)

# Thread pool untuk blocking I/O (model inference, DB queries, LLM calls)
_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="agent-worker")


# ═══════════════════════════════════════════
# INTERNAL HELPERS
# ═══════════════════════════════════════════

def _preprocess_file_context(file_context: Optional[str]) -> Optional[str]:
    """Defensive preprocessing untuk file context dari upload."""
    if not file_context or not file_context.strip():
        return None

    cleaned = file_context.strip()

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


def _retrieve_context(intent: str, query: str, limit: int = 5, graph_limit: int = 4, persona: str = "umum") -> str:
    """Memilih dan menjalankan retriever berdasarkan intent yang terdeteksi dengan profil per-persona."""
    retriever_map: dict[str, Any] = {
        "konsultasi": content_based_recommendation,
        "ensiklopedia": search_encyclopedia,
        "edukasi": retrieve_education_corpus,
    }

    retriever = retriever_map.get(intent)
    if retriever:
        return retriever(query, limit=limit, graph_limit=graph_limit, persona=persona)

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
    model_tier: Optional[str] = None,
) -> dict[str, Any]:
    """Pipeline sinkron untuk memproses query user."""
    intent = classify_intent(query)
    logger.info(
        f"Intent classified: '{intent}' | "
        f"Query: '{query[:80]}...' | Mode: {ai_mode} | Tier: {model_tier}"
    )

    # 1. Resolve model route & registry configs
    resolved_tier = ModelTier.FAST
    if model_tier:
        resolved_tier = ModelTier.THINKING if str(model_tier).lower() == "thinking" else ModelTier.FAST
    elif model:
        resolved_tier = ModelTier.THINKING if model == settings.MODEL_THINKING else ModelTier.FAST

    route = resolve_model(resolved_tier, model)
    registry_conf = MODEL_REGISTRY[route.model_tier]

    limit = registry_conf["retrieval_limit"]
    graph_limit = registry_conf["graph_limit"]

    # ── CAPTURE UPLOADED FILE PAYLOAD DATA ──
    file_data_content = _preprocess_file_context(file_context)
    if file_data_content:
        logger.info(
            f"File context captured: {len(file_data_content)} chars "
            f"(will be injected into LLM system prompt)"
        )

    # ── QUIZ INTENT: Agentic Tool-Calling ──
    if intent == "generate_quiz":
        try:
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                quiz_data = loop.run_until_complete(
                    generate_interactive_quiz_tool(
                        topic=query,
                        jumlah_soal=3,
                        ai_mode=ai_mode,
                        file_context=file_data_content,
                        model=route.used_model,
                    )
                )
            finally:
                loop.close()
            return {
                "intent_detected": "quiz_rendered",
                "quiz_payload": quiz_data,
                "ai_response": (
                    "Berikut adalah kuis interaktif Anda "
                    "berdasarkan materi yang tersedia."
                ),
                "model_route": route,
            }
        except Exception as e:
            logger.error(f"Quiz generation failed: {e}", exc_info=True)
            return {
                "intent_detected": "generate_quiz",
                "ai_response": (
                    f"Maaf, gagal membuat kuis. "
                    f"Silakan coba lagi. ({type(e).__name__})"
                ),
                "model_route": route,
            }

    # ── RAG INTENT: Retrieval + LLM Formatting ──
    search_query = query
    if file_data_content:
        search_query = f"{query} {file_data_content.strip()}"

    context_data = _retrieve_context(intent, search_query, limit=limit, graph_limit=graph_limit, persona=ai_mode)

    if (not context_data or context_data.strip() == "" or "0 records found" in context_data) and file_data_content:
        logger.info("Graph search was empty but file context exists. Feeding vision data into context_data to bypass strict RAG constraints.")
        context_data = f"Data Hasil Analisis Gambar/Berkas Laboratorium: {file_data_content}"

    final_response = generate_strict_response(
        query=query,
        context=context_data,
        ai_mode=ai_mode,
        intent=intent,
        file_context=file_data_content,
        model=route.used_model,
        model_tier=route.model_tier.value,
    )

    return {
        "intent_detected": intent,
        "ai_response": final_response,
        "model_route": route,
    }


async def process_user_query(
    query: str,
    ai_mode: str,
    file_context: Optional[str] = None,
    model: Optional[str] = None,
    model_tier: Optional[str] = None,
) -> dict[str, Any]:
    """Pipeline utama (non-streaming, async) untuk memproses query user."""
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        _executor,
        _process_query_sync,
        query,
        ai_mode,
        file_context,
        model,
        model_tier,
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
    model_tier: Optional[str] = None,
) -> Generator[dict[str, Any], None, None]:
    """Pipeline sinkron streaming untuk SSE (dijalankan di thread pool)."""
    intent = classify_intent(query)
    logger.info(f"[Stream] Intent: '{intent}' | Query: '{query[:80]}...' | Mode: {ai_mode} | Tier: {model_tier}")

    # 1. Resolve model route & configs
    resolved_tier = ModelTier.FAST
    if model_tier:
        resolved_tier = ModelTier.THINKING if str(model_tier).lower() == "thinking" else ModelTier.FAST
    elif model:
        resolved_tier = ModelTier.THINKING if model == settings.MODEL_THINKING else ModelTier.FAST

    route = resolve_model(resolved_tier, model)
    registry_conf = MODEL_REGISTRY[route.model_tier]

    limit = registry_conf["retrieval_limit"]
    graph_limit = registry_conf["graph_limit"]

    # Yield intent & model route metadata
    yield {"event": "intent", "data": intent}
    yield {"event": "model_route", "data": route.dict()}

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
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                quiz_data = loop.run_until_complete(
                    generate_interactive_quiz_tool(
                        topic=query,
                        jumlah_soal=3,
                        ai_mode=ai_mode,
                        file_context=file_data_content,
                        model=route.used_model,
                    )
                )
            finally:
                loop.close()
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

    context_data = _retrieve_context(intent, search_query, limit=limit, graph_limit=graph_limit, persona=ai_mode)

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
            model=route.used_model,
            model_tier=route.model_tier.value,
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
    model_tier: Optional[str] = None,
) -> AsyncGenerator[dict[str, Any], None]:
    """Pipeline streaming async untuk SSE endpoint."""
    loop = asyncio.get_event_loop()
    import queue

    event_queue: queue.Queue[dict[str, Any] | None] = queue.Queue()

    def _run_stream() -> None:
        """Worker: jalankan stream sync dan masukkan event ke queue."""
        try:
            for event in _stream_query_sync(query, ai_mode, file_context, model, model_tier):
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
