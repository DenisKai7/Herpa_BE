"""
Chat API - Endpoints untuk chat messaging dan manajemen sesi chat.

Fitur:
- Blocking message: Response AI lengkap sekaligus.
- Streaming message: Server-Sent Events (SSE) token-by-token.
- Manajemen chat: rename, pin, share, delete, list, history.
- Public shared chat: read-only tanpa autentikasi.

Keamanan:
- Semua endpoint (kecuali public) dilindungi oleh JWT verification.
- User hanya bisa mengakses chat miliknya sendiri (ownership check).
"""

import json
import logging
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse

from app.agent.orchestrator import process_user_query, process_user_query_stream
from app.core.database import supabase
from app.core.dependencies import verify_user
from app.models.schemas import ChatActionRequest, ChatRequest, ChatResponse

logger = logging.getLogger(__name__)
router = APIRouter()


# ═══════════════════════════════════════════
# HELPER: Ownership Verification
# ═══════════════════════════════════════════

def _verify_chat_ownership(chat_id: str, user_id: str) -> dict[str, Any]:
    """
    Memverifikasi bahwa chat_id milik user_id.

    Args:
        chat_id: UUID chat session.
        user_id: UUID user yang terautentikasi.

    Returns:
        Data chat record jika valid.

    Raises:
        HTTPException 404: Jika chat tidak ditemukan.
        HTTPException 403: Jika chat bukan milik user.
    """
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
) -> str:
    """
    Menyimpan pesan user dan AI response ke database.

    Jika chat_id belum ada, buat chat session baru terlebih dahulu.

    Args:
        chat_id: UUID chat session (None = buat baru).
        user_id: UUID user pemilik chat.
        user_message: Pesan asli dari pengguna.
        ai_content: Respons AI (teks atau JSON quiz).
        is_quiz: True jika response adalah quiz payload.
        intent: Intent yang terdeteksi oleh NLU.

    Returns:
        chat_id (str): UUID chat session (baru atau existing).
    """
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

    supabase.table("messages").insert([
        {
            "chat_id": chat_id,
            "role": "user",
            "content": user_message,
        },
        {
            "chat_id": chat_id,
            "role": "ai",
            "content": ai_content,
            "metadata": {"is_quiz": is_quiz, "intent": intent},
        },
    ]).execute()

    supabase.table("chats").update(
        {"updated_at": "now()"}
    ).eq("id", chat_id).execute()

    return chat_id


# ═══════════════════════════════════════════
# CHAT MESSAGE ENDPOINTS
# ═══════════════════════════════════════════

@router.post(
    "/message",
    response_model=ChatResponse,
    summary="Kirim pesan ke AI (blocking)",
)
async def chat_endpoint(
    req: ChatRequest,
    user_id: str = Depends(verify_user),
) -> ChatResponse:
    """
    Endpoint utama untuk mengirim pesan ke AI agent (mode blocking).

    Pipeline:
    1. Verifikasi JWT token via dependency.
    2. Proses query melalui Agentic Pipeline (intent -> retrieval -> LLM).
    3. Simpan pesan user dan AI response ke database.
    4. Return structured response.

    Args:
        req: ChatRequest berisi message, ai_mode, dan opsional file_context/chat_id.
        user_id: UUID user dari JWT (injected by Depends).

    Returns:
        ChatResponse berisi chat_id, intent, response, dan opsional quiz_data.
    """
    try:
        # Jalankan agentic pipeline
        pipeline_result = await process_user_query(
            query=req.message,
            ai_mode=req.ai_mode,
            file_context=req.file_context,
        )

        is_quiz = pipeline_result.get("intent_detected") == "quiz_rendered"
        ai_response_text = pipeline_result["ai_response"]
        db_content = (
            json.dumps(pipeline_result["quiz_payload"])
            if is_quiz
            else ai_response_text
        )

        # Simpan ke database
        chat_id = _save_messages_to_db(
            chat_id=req.chat_id,
            user_id=user_id,
            user_message=req.message,
            ai_content=db_content,
            is_quiz=is_quiz,
            intent=pipeline_result["intent_detected"],
        )

        return ChatResponse(
            chat_id=chat_id,
            intent=pipeline_result["intent_detected"],
            response=ai_response_text,
            quiz_data=pipeline_result.get("quiz_payload"),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/message/stream", summary="Kirim pesan ke AI (SSE streaming)")
async def chat_stream_endpoint(
    req: ChatRequest,
    user_id: str = Depends(verify_user),
) -> StreamingResponse:
    """
    Endpoint streaming menggunakan Server-Sent Events (SSE).

    Events yang di-emit:
    - intent: intent yang terdeteksi oleh NLU.
    - token: setiap chunk teks dari LLM response.
    - quiz: payload quiz lengkap (jika intent = generate_quiz).
    - full_response: response lengkap (untuk disimpan ke DB).
    - error: pesan error jika terjadi kegagalan.
    - done: streaming selesai, berisi chat_id.

    Args:
        req: ChatRequest berisi message, ai_mode, dan opsional file_context/chat_id.
        user_id: UUID user dari JWT (injected by Depends).

    Returns:
        StreamingResponse dengan media_type text/event-stream.
    """
    async def event_generator():
        full_ai_response = ""
        chat_id = req.chat_id
        is_quiz = False
        quiz_payload: Optional[dict[str, Any]] = None
        detected_intent = ""

        try:
            async for event in process_user_query_stream(
                query=req.message,
                ai_mode=req.ai_mode,
                file_context=req.file_context,
            ):
                event_type = event["event"]
                event_data = event["data"]

                if event_type == "intent":
                    detected_intent = event_data
                    yield f"event: intent\ndata: {json.dumps({'intent': event_data})}\n\n"

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
                    yield f"event: error\ndata: {json.dumps({'error': event_data})}\n\n"

                elif event_type == "done":
                    # Simpan ke database setelah streaming selesai
                    try:
                        db_content = (
                            json.dumps(quiz_payload)
                            if is_quiz
                            else full_ai_response
                        )
                        chat_id = _save_messages_to_db(
                            chat_id=chat_id,
                            user_id=user_id,
                            user_message=req.message,
                            ai_content=db_content,
                            is_quiz=is_quiz,
                            intent=detected_intent,
                        )
                    except Exception as db_err:
                        logger.error(
                            f"DB save error during stream: {db_err}",
                            exc_info=True,
                        )

                    yield f"event: done\ndata: {json.dumps({'chat_id': chat_id})}\n\n"

        except Exception as e:
            logger.error(f"Stream error: {e}", exc_info=True)
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ═══════════════════════════════════════════
# CHAT LIST & HISTORY ENDPOINTS
# ═══════════════════════════════════════════

@router.get("/list", summary="Daftar chat user")
async def list_user_chats(
    user_id: str = Depends(verify_user),
) -> dict[str, Any]:
    """
    Mengambil daftar semua chat session milik user yang terautentikasi.

    Diurutkan berdasarkan status pinned (terlebih dahulu) dan waktu update terbaru.

    Args:
        user_id: UUID user dari JWT (injected by Depends).

    Returns:
        Dict berisi list chats.
    """
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
    """
    Mengambil seluruh pesan dalam satu chat session, diurutkan kronologis.

    Memverifikasi ownership: hanya pemilik chat yang bisa melihat riwayat.

    Args:
        chat_id: UUID chat session.
        user_id: UUID user dari JWT (injected by Depends).

    Returns:
        Dict berisi list messages.
    """
    # Verifikasi kepemilikan
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


# ═══════════════════════════════════════════
# CHAT MANAGEMENT ENDPOINTS
# ═══════════════════════════════════════════

@router.patch("/{chat_id}/rename", summary="Ganti judul chat")
async def rename_chat(
    chat_id: str,
    req: ChatActionRequest,
    user_id: str = Depends(verify_user),
) -> dict[str, Any]:
    """
    Mengubah judul chat session.

    Args:
        chat_id: UUID chat session.
        req: ChatActionRequest berisi field 'title' baru.
        user_id: UUID user dari JWT (injected by Depends).

    Returns:
        Dict konfirmasi dengan judul baru.
    """
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
    """
    Toggle pin status sebuah chat agar muncul di atas sidebar.

    Args:
        chat_id: UUID chat session.
        req: ChatActionRequest berisi field 'is_pinned'.
        user_id: UUID user dari JWT (injected by Depends).

    Returns:
        Dict konfirmasi dengan status pin terbaru.
    """
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
    """
    Toggle public visibility. Jika is_public=true, chat bisa diakses via public URL.

    Args:
        chat_id: UUID chat session.
        req: ChatActionRequest berisi field 'is_public'.
        user_id: UUID user dari JWT (injected by Depends).

    Returns:
        Dict konfirmasi dengan status share dan opsional public_url.
    """
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
    """
    Endpoint publik untuk membaca chat yang di-share.

    Tidak memerlukan autentikasi. Hanya menampilkan chat yang
    memiliki is_public=true.

    Args:
        chat_id: UUID chat session.

    Returns:
        Dict berisi title, created_at, dan list messages.
    """
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
    """
    Menghapus chat session dan semua messages di dalamnya.

    Memverifikasi ownership sebelum menghapus.

    Args:
        chat_id: UUID chat session.
        user_id: UUID user dari JWT (injected by Depends).

    Returns:
        Dict konfirmasi penghapusan.
    """
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
