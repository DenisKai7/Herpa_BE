"""
Chat API - Endpoints untuk chat messaging dan manajemen sesi chat.
"""

import json
import logging
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse

from app.agent.orchestrator import process_user_query, process_user_query_stream
from app.core.database import supabase
from app.core.dependencies import verify_user, verify_user_with_role, resolve_model_for_role, resolve_model, ModelTier
from app.models.schemas import ChatActionRequest, ChatRequest, ChatResponse
from app.core.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# HELPER: Error Classification
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def classify_provider_error(exc: Exception) -> str:
    message = str(exc).lower()

    if "model_not_supported" in message:
        return "model_not_supported"

    if "not supported by any provider" in message:
        return "model_not_supported"

    if "429" in message or "rate limit" in message:
        return "rate_limited"

    if "timeout" in message:
        return "timeout"

    if "401" in message:
        return "authentication_failed"

    if "403" in message:
        return "access_denied"

    return "provider_error"


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# HELPER: Ownership Verification
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def _verify_chat_ownership(chat_id: str, user_id: str) -> dict[str, Any]:
    """Memverifikasi bahwa chat_id milik user_id."""
    try:
        result = (
            supabase.table("chats")
            .select("id, user_id, title, is_pinned, is_public")
            .eq("id", chat_id)
            .execute()
        )
    except Exception as e:
        logger.error(f"Chat ownership check DB error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Gagal memverifikasi kepemilikan chat.")

    if not result.data:
        raise HTTPException(status_code=404, detail="Chat tidak ditemukan.")

    chat_record = result.data[0]
    if chat_record["user_id"] != user_id:
        raise HTTPException(status_code=403, detail="Anda tidak memiliki akses ke chat ini.")

    return chat_record


def _save_messages_to_db(
    chat_id: str,
    user_id: str,
    user_message: str,
    ai_content: str,
    is_quiz: bool,
    intent: str,
    file_url: Optional[str] = None,
    file_name: Optional[str] = None,
    file_type: Optional[str] = None,
    execution_metadata: Optional[dict[str, Any]] = None,
) -> str:
    """Menyimpan pesan user dan AI response ke database."""
    if not chat_id:
        chat_data = (
            supabase.table("chats")
            .insert({
                "user_id": user_id,
                "title": user_message[:50].strip(),
            })
            .execute()
        )
        chat_id = chat_data.data[0]["id"]
        logger.info(f"New chat session created: {chat_id}")

    user_meta = {}
    if file_url:
        user_meta = {
            "file_url": file_url,
            "file_name": file_name,
            "file_type": file_type
        }

    supabase.table("messages").insert([
        {
            "chat_id": chat_id,
            "role": "user",
            "content": user_message,
            "metadata": user_meta,
        },
        {
            "chat_id": chat_id,
            "role": "ai",
            "content": ai_content,
            "metadata": {
                "is_quiz": is_quiz,
                "intent": intent,
                **(execution_metadata or {})
            },
        },
    ]).execute()

    supabase.table("chats").update(
        {"updated_at": "now()"}
    ).eq("id", chat_id).execute()

    return chat_id

def _resolve_attachment_file_context(req: ChatRequest, user_id: str) -> tuple[Optional[str], dict[str, Any]]:
    """Resolve legacy file_context or session-scoped uploaded attachment context."""
    if req.file_context:
        return req.file_context, {}
    if not req.attachment_id:
        return None, {}
    from app.api.upload import get_attachment_context_for_user

    payload = get_attachment_context_for_user(user_id, req.attachment_id)
    if not payload:
        raise HTTPException(status_code=404, detail="Attachment tidak ditemukan atau bukan milik user ini.")

    processing_status = payload.get("processing_status", "completed" if payload.get("formatted_context") else "queued")
    if processing_status != "completed":
        code = "ATTACHMENT_PROCESSING_FAILED" if processing_status == "failed" else "ATTACHMENT_NOT_READY"
        message = (
            "Lampiran gagal dianalisis. Silakan coba lagi atau hapus lampiran."
            if processing_status == "failed"
            else "Lampiran masih diproses. Silakan tunggu hingga selesai."
        )
        raise HTTPException(
            status_code=409,
            detail={
                "success": False,
                "error": {
                    "code": code,
                    "message": message,
                    "retryable": True,
                    "processing_status": processing_status,
                },
            },
        )

    formatted_context = payload.get("formatted_context")
    if not formatted_context:
        raise HTTPException(
            status_code=409,
            detail={
                "success": False,
                "error": {
                    "code": "ATTACHMENT_NOT_READY",
                    "message": "Lampiran masih diproses. Silakan tunggu hingga selesai.",
                    "retryable": True,
                    "processing_status": processing_status,
                },
            },
        )

    analysis = payload.get("analysis") or {}
    return formatted_context, {
        "attachment_id": req.attachment_id,
        "attachment_filename": payload.get("filename"),
        "attachment_preview_url": payload.get("preview_url"),
        "attachment_verification_status": analysis.get("verification_status"),
        "attachment_confidence": analysis.get("confidence"),
    }


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# CHAT MESSAGE ENDPOINTS
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

@router.post(
    "/message",
    response_model=ChatResponse,
    summary="Kirim pesan ke AI (blocking)",
)
async def chat_endpoint(
    req: ChatRequest,
    user_data: dict[str, str] = Depends(verify_user_with_role),
) -> ChatResponse:
    """Endpoint utama untuk mengirim pesan ke AI agent (mode blocking)."""
    user_id = user_data["user_id"]
    user_role = user_data["role"]

    resolved_file_context, attachment_metadata = _resolve_attachment_file_context(req, user_id)

    # 1. Resolve persona & model tier (support backward compatibility)
    persona_str = req.persona or req.ai_mode or "umum"
    # Resolve ModelTier
    model_tier_str = req.model_tier
    if not model_tier_str:
        if req.model_choice:
            if req.model_choice.lower() in ("fast", "thinking"):
                model_tier_str = req.model_choice.lower()
            else:
                model_tier_str = "thinking" if req.model_choice == settings.MODEL_THINKING else "fast"
        else:
            model_tier_str = "thinking" if user_role in ("tenaga_medis", "peneliti") else "fast"

    # 2. Run agentic pipeline with fallback logic
    fallback_used = False
    used_model = settings.MODEL_FAST
    resolved_model_tier = model_tier_str
    pipeline_result = None

    try:
        pipeline_result = await process_user_query(
            query=req.message,
            ai_mode=persona_str,
            file_context=resolved_file_context,
            model=req.model_choice,
            model_tier=model_tier_str,
        )
        route = pipeline_result.get("model_route")
        if route:
            used_model = route.used_model
            resolved_model_tier = route.model_tier.value
            fallback_used = route.fallback_used
    except Exception as e:
        error_type = classify_provider_error(e)
        logger.warning(f"Primary model execution failed: {e}. Error type: {error_type}")

        # Try fallback if allowed
        if settings.ALLOW_MODEL_FALLBACK:
            fallback_tier = "thinking" if model_tier_str == "fast" else "fast"
            fallback_model = settings.MODEL_THINKING if fallback_tier == "thinking" else settings.MODEL_FAST
            logger.info(f"Attempting fallback to model tier: {fallback_tier} ({fallback_model})")
            try:
                pipeline_result = await process_user_query(
                    query=req.message,
                    ai_mode=persona_str,
                    file_context=resolved_file_context,
                    model=fallback_model,
                    model_tier=fallback_tier,
                )
                fallback_used = True
                route = pipeline_result.get("model_route")
                if route:
                    used_model = route.used_model
                    resolved_model_tier = route.model_tier.value
            except Exception as fallback_exc:
                logger.error(f"Fallback model also failed: {fallback_exc}", exc_info=True)
                raise HTTPException(
                    status_code=503,
                    detail={
                        "success": False,
                        "error": {
                            "code": "LLM_TEMPORARILY_UNAVAILABLE",
                            "message": "Layanan AI sedang tidak tersedia. Silakan coba kembali beberapa saat lagi.",
                            "retryable": True
                        }
                    }
                )
        else:
            raise HTTPException(
                status_code=503,
                detail={
                    "success": False,
                    "error": {
                        "code": "LLM_TEMPORARILY_UNAVAILABLE",
                        "message": "Layanan AI sedang tidak tersedia. Silakan coba kembali beberapa saat lagi.",
                        "retryable": True
                    }
                }
            )

    if not pipeline_result:
        raise HTTPException(
            status_code=503,
            detail={
                "success": False,
                "error": {
                    "code": "LLM_TEMPORARILY_UNAVAILABLE",
                    "message": "Layanan AI sedang tidak tersedia. Silakan coba kembali beberapa saat lagi.",
                    "retryable": True
                }
            }
        )

    is_quiz = pipeline_result.get("intent_detected") == "quiz_rendered"
    ai_response_text = pipeline_result["ai_response"]
    db_content = (
        json.dumps(pipeline_result["quiz_payload"])
        if is_quiz
        else ai_response_text
    )

    execution_metadata = {
        "persona": persona_str,
        "model_tier": resolved_model_tier,
        "requested_model": req.model_choice,
        "used_model": used_model,
        "provider": "hf_router",
        "fallback_used": fallback_used,
        "retrieval_used": True,
        "evidence_level": "mixed",
        **attachment_metadata
    }

    # Simpan ke database
    chat_id = _save_messages_to_db(
        chat_id=req.chat_id,
        user_id=user_id,
        user_message=req.message,
        ai_content=db_content,
        is_quiz=is_quiz,
        intent=pipeline_result["intent_detected"],
        file_url=req.file_url,
        file_name=req.file_name,
        file_type=req.file_type,
        execution_metadata=execution_metadata,
    )

    return ChatResponse(
        chat_id=chat_id,
        intent=pipeline_result["intent_detected"],
        response=ai_response_text,
        quiz_data=pipeline_result.get("quiz_payload"),
        metadata=execution_metadata,
    )


@router.post("/message/stream", summary="Kirim pesan ke AI (SSE streaming)")
async def chat_stream_endpoint(
    req: ChatRequest,
    user_data: dict[str, str] = Depends(verify_user_with_role),
) -> StreamingResponse:
    """Endpoint streaming menggunakan Server-Sent Events (SSE)."""
    user_id = user_data["user_id"]
    user_role = user_data["role"]

    resolved_file_context, attachment_metadata = _resolve_attachment_file_context(req, user_id)

    # Resolve Model & Tier
    persona_str = req.persona or req.ai_mode or "umum"
    model_tier_str = req.model_tier
    if not model_tier_str:
        if req.model_choice:
            if req.model_choice.lower() in ("fast", "thinking"):
                model_tier_str = req.model_choice.lower()
            else:
                model_tier_str = "thinking" if req.model_choice == settings.MODEL_THINKING else "fast"
        else:
            model_tier_str = "thinking" if user_role in ("tenaga_medis", "peneliti") else "fast"

    async def event_generator():
        full_ai_response = ""
        chat_id = req.chat_id
        is_quiz = False
        quiz_payload: Optional[dict[str, Any]] = None
        detected_intent = ""
        used_model = settings.MODEL_FAST
        resolved_model_tier = model_tier_str
        fallback_used = False

        try:
            async for event in process_user_query_stream(
                query=req.message,
                ai_mode=persona_str,
                file_context=resolved_file_context,
                model=req.model_choice,
                model_tier=model_tier_str,
            ):
                event_type = event["event"]
                event_data = event["data"]

                if event_type == "intent":
                    detected_intent = event_data
                    yield f"event: intent\ndata: {json.dumps({'intent': event_data})}\n\n"

                elif event_type == "model_route":
                    used_model = event_data.get("used_model", used_model)
                    resolved_model_tier = event_data.get("model_tier", resolved_model_tier)
                    fallback_used = event_data.get("fallback_used", fallback_used)

                elif event_type == "token":
                    full_ai_response += event_data
                    yield f"event: token\ndata: {json.dumps({'token': event_data})}\n\n"

                elif event_type == "quiz":
                    is_quiz = True
                    quiz_payload = event_data.get("quiz_payload")
                    full_ai_response = event_data.get("message", "")
                    yield f"event: quiz\ndata: {json.dumps(event_data)}\n\n"

                elif event_type == "full_response":
                    full_ai_response = event_data

                elif event_type == "error":
                    # Map to a friendly user error
                    friendly_error = {
                        "success": False,
                        "error": {
                            "code": "LLM_TEMPORARILY_UNAVAILABLE",
                            "message": "Layanan AI sedang tidak tersedia. Silakan coba kembali beberapa saat lagi.",
                            "retryable": True
                        }
                    }
                    yield f"event: error\ndata: {json.dumps(friendly_error)}\n\n"

                elif event_type == "done":
                    # Simpan ke database setelah streaming selesai
                    try:
                        db_content = (
                            json.dumps(quiz_payload)
                            if is_quiz
                            else full_ai_response
                        )
                        execution_metadata = {
                            "persona": persona_str,
                            "model_tier": resolved_model_tier,
                            "requested_model": req.model_choice,
                            "used_model": used_model,
                            "provider": "hf_router",
                            "fallback_used": fallback_used,
                            "retrieval_used": True,
                            "evidence_level": "mixed",
                            **attachment_metadata
                        }
                        chat_id = _save_messages_to_db(
                            chat_id=chat_id,
                            user_id=user_id,
                            user_message=req.message,
                            ai_content=db_content,
                            is_quiz=is_quiz,
                            intent=detected_intent,
                            file_url=req.file_url,
                            file_name=req.file_name,
                            file_type=req.file_type,
                            execution_metadata=execution_metadata,
                        )
                    except Exception as db_err:
                        logger.error(
                            f"DB save error during stream: {db_err}",
                            exc_info=True,
                        )

                    yield f"event: done\ndata: {json.dumps({'chat_id': chat_id})}\n\n"

        except Exception as e:
            logger.error(f"Stream error: {e}", exc_info=True)
            friendly_error = {
                "success": False,
                "error": {
                    "code": "LLM_TEMPORARILY_UNAVAILABLE",
                    "message": "Layanan AI sedang tidak tersedia. Silakan coba kembali beberapa saat lagi.",
                    "retryable": True
                }
            }
            yield f"event: error\ndata: {json.dumps(friendly_error)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# CHAT LIST & HISTORY ENDPOINTS
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

@router.get("/list", summary="Daftar chat user")
async def list_user_chats(
    user_id: str = Depends(verify_user),
) -> dict[str, Any]:
    """Mengambil daftar semua chat session milik user."""
    try:
        result = (
            supabase.table("chats")
            .select("id, title, is_pinned, is_public, created_at, updated_at")
            .eq("user_id", user_id)
            .order("is_pinned", desc=True)
            .order("updated_at", desc=True)
            .execute()
        )
        return {"chats": result.data or []}
    except Exception as e:
        logger.error(f"List chats error for user {user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{chat_id}/messages", summary="Riwayat pesan dalam satu chat")
async def get_chat_messages(
    chat_id: str,
    user_id: str = Depends(verify_user),
) -> dict[str, Any]:
    """Mengambil seluruh pesan dalam satu chat session."""
    _verify_chat_ownership(chat_id, user_id)

    try:
        messages = (
            supabase.table("messages")
            .select("id, role, content, metadata, created_at")
            .eq("chat_id", chat_id)
            .order("created_at", desc=False)
            .execute()
        )
        return {"messages": messages.data or []}
    except Exception as e:
        logger.error(f"Get messages error for chat {chat_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# CHAT MANAGEMENT ENDPOINTS
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

@router.patch("/{chat_id}/rename", summary="Ganti judul chat")
async def rename_chat(
    chat_id: str,
    req: ChatActionRequest,
    user_id: str = Depends(verify_user),
) -> dict[str, Any]:
    """Mengubah judul chat session."""
    if not req.title:
        raise HTTPException(
            status_code=400,
            detail="Field 'title' wajib diisi untuk rename.",
        )

    _verify_chat_ownership(chat_id, user_id)

    try:
        supabase.table("chats").update(
            {"title": req.title}
        ).eq("id", chat_id).execute()
        return {"message": "Judul chat berhasil diubah.", "new_title": req.title}
    except Exception as e:
        logger.error(f"Rename chat error for {chat_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/{chat_id}/pin", summary="Pin/unpin chat")
async def toggle_pin_chat(
    chat_id: str,
    req: ChatActionRequest,
    user_id: str = Depends(verify_user),
) -> dict[str, Any]:
    """Toggle pin status sebuah chat."""
    if req.is_pinned is None:
        raise HTTPException(
            status_code=400,
            detail="Field 'is_pinned' wajib diisi.",
        )

    _verify_chat_ownership(chat_id, user_id)

    try:
        supabase.table("chats").update(
            {"is_pinned": req.is_pinned}
        ).eq("id", chat_id).execute()
        status = "disemat" if req.is_pinned else "batal disemat"
        return {"message": f"Chat berhasil {status}.", "is_pinned": req.is_pinned}
    except Exception as e:
        logger.error(f"Pin chat error for {chat_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/{chat_id}/share", summary="Share/unshare chat (public link)")
async def toggle_share_chat(
    chat_id: str,
    req: ChatActionRequest,
    user_id: str = Depends(verify_user),
) -> dict[str, Any]:
    """Toggle public visibility."""
    if req.is_public is None:
        raise HTTPException(
            status_code=400,
            detail="Field 'is_public' wajib diisi.",
        )

    _verify_chat_ownership(chat_id, user_id)

    try:
        supabase.table("chats").update(
            {"is_public": req.is_public}
        ).eq("id", chat_id).execute()
        public_url = f"/api/chat/public/{chat_id}" if req.is_public else None
        return {
            "message": "Status share diperbarui.",
            "is_public": req.is_public,
            "public_url": public_url,
        }
    except Exception as e:
        logger.error(f"Share chat error for {chat_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/public/{chat_id}", summary="Lihat shared chat (read-only)")
async def get_public_chat(chat_id: str) -> dict[str, Any]:
    """Endpoint publik untuk membaca chat yang di-share."""
    try:
        chat_info = (
            supabase.table("chats")
            .select("is_public, title, created_at")
            .eq("id", chat_id)
            .execute()
        )

        if not chat_info.data or not chat_info.data[0].get("is_public"):
            raise HTTPException(
                status_code=403,
                detail="Chat ini bersifat privat atau tidak ditemukan.",
            )

        messages = (
            supabase.table("messages")
            .select("role, content, metadata, created_at")
            .eq("chat_id", chat_id)
            .order("created_at", desc=False)
            .execute()
        )

        return {
            "title": chat_info.data[0]["title"],
            "created_at": chat_info.data[0].get("created_at"),
            "messages": messages.data or [],
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get public chat error for {chat_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{chat_id}", summary="Hapus chat beserta semua pesannya")
async def delete_chat(
    chat_id: str,
    user_id: str = Depends(verify_user),
) -> dict[str, str]:
    """Menghapus chat session dan semua messages di dalamnya."""
    _verify_chat_ownership(chat_id, user_id)

    try:
        # Hapus messages terlebih dahulu (FK constraint)
        supabase.table("messages").delete().eq("chat_id", chat_id).execute()
        supabase.table("chats").delete().eq("id", chat_id).execute()

        logger.info(f"Chat {chat_id} deleted by user {user_id}")
        return {"message": "Chat berhasil dihapus."}
    except Exception as e:
        logger.error(f"Delete chat error for {chat_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


