"""
Admin API - Dashboard analytics & user management.

Semua endpoint dilindungi oleh verify_admin dependency yang memvalidasi:
1. JWT token valid (Supabase Auth).
2. User memiliki role 'admin' di tabel profiles.

Fitur:
- Dashboard analytics (total users, chats, messages).
- Activity analytics per periode waktu.
- User management (list, role update, soft-delete).
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from app.core.database import supabase
from app.core.dependencies import verify_admin
from app.models.schemas import AnalyticsResponse, RoleUpdateRequest

logger = logging.getLogger(__name__)
router = APIRouter()


# ═══════════════════════════════════════════
# ANALYTICS ENDPOINTS
# ═══════════════════════════════════════════

@router.get(
    "/analytics",
    response_model=AnalyticsResponse,
    summary="Dashboard analytics",
)
async def get_dashboard_analytics(
    admin_id: str = Depends(verify_admin),
) -> AnalyticsResponse:
    """
    Mengambil statistik dashboard:
    - Total users terdaftar.
    - Total chat sessions.
    - Total messages.

    Args:
        admin_id: UUID admin dari JWT (injected by Depends).

    Returns:
        AnalyticsResponse dengan data statistik.
    """
    try:
        users = supabase.table("profiles").select("id", count="exact").execute()
        chats = supabase.table("chats").select("id", count="exact").execute()
        messages = supabase.table("messages").select("id", count="exact").execute()

        return AnalyticsResponse(
            total_users=users.count or 0,
            total_chat_sessions=chats.count or 0,
            total_messages=messages.count or 0,
            status="Healthy",
        )

    except Exception as e:
        logger.error(f"Analytics error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Gagal mengambil data analytics: {str(e)}",
        )


@router.get("/analytics/activity", summary="Activity analytics over time")
async def get_activity_analytics(
    days: int = 7,
    admin_id: str = Depends(verify_admin),
) -> dict[str, Any]:
    """
    Mengambil statistik aktivitas dalam N hari terakhir.
    Berguna untuk chart/grafik di dashboard admin.

    Args:
        days: Jumlah hari ke belakang (default: 7).
        admin_id: UUID admin dari JWT (injected by Depends).

    Returns:
        Dict berisi period_days, chats_count, messages_count.
    """
    try:
        start_date = (
            datetime.now(timezone.utc) - timedelta(days=days)
        ).isoformat()

        recent_chats = (
            supabase.table("chats")
            .select("id, created_at")
            .gte("created_at", start_date)
            .execute()
        )

        recent_messages = (
            supabase.table("messages")
            .select("id, created_at")
            .gte("created_at", start_date)
            .execute()
        )

        return {
            "period_days": days,
            "chats_count": len(recent_chats.data) if recent_chats.data else 0,
            "messages_count": len(recent_messages.data) if recent_messages.data else 0,
        }

    except Exception as e:
        logger.error(f"Activity analytics error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════
# USER MANAGEMENT ENDPOINTS
# ═══════════════════════════════════════════

@router.get("/users", summary="Daftar semua user")
async def list_users(
    page: int = 1,
    limit: int = 20,
    admin_id: str = Depends(verify_admin),
) -> dict[str, Any]:
    """
    Mengambil daftar user dengan pagination.

    Args:
        page: Nomor halaman (mulai dari 1).
        limit: Jumlah item per halaman (default: 20).
        admin_id: UUID admin dari JWT (injected by Depends).

    Returns:
        Dict berisi data (list users), total, page, limit.
    """
    try:
        offset = (page - 1) * limit
        result = (
            supabase.table("profiles")
            .select("*", count="exact")
            .order("created_at", desc=True)
            .range(offset, offset + limit - 1)
            .execute()
        )

        return {
            "data": result.data or [],
            "total": result.count or 0,
            "page": page,
            "limit": limit,
        }

    except Exception as e:
        logger.error(f"List users error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/users/role", summary="Ubah role user")
async def update_user_role(
    req: RoleUpdateRequest,
    admin_id: str = Depends(verify_admin),
) -> dict[str, Any]:
    """
    Mengubah role user (admin/user).

    Proteksi: Admin tidak bisa mengubah role dirinya sendiri
    untuk mencegah accidental privilege removal.

    Args:
        req: RoleUpdateRequest berisi target_user_id dan new_role.
        admin_id: UUID admin dari JWT (injected by Depends).

    Returns:
        Dict konfirmasi dengan target_user_id dan new_role.
    """
    if req.target_user_id == admin_id:
        raise HTTPException(
            status_code=400,
            detail="Tidak dapat mengubah role diri sendiri.",
        )

    try:
        # Verifikasi target user ada
        target = (
            supabase.table("profiles")
            .select("id, role")
            .eq("id", req.target_user_id)
            .execute()
        )
        if not target.data:
            raise HTTPException(status_code=404, detail="User tidak ditemukan.")

        supabase.table("profiles").update(
            {"role": req.new_role}
        ).eq("id", req.target_user_id).execute()

        logger.info(
            f"Admin {admin_id} changed role of {req.target_user_id} "
            f"to {req.new_role}"
        )
        return {
            "message": f"Berhasil mengubah role user menjadi '{req.new_role}'.",
            "target_user_id": req.target_user_id,
            "new_role": req.new_role,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Update role error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/users/{user_id}", summary="Hapus user (soft delete)")
async def delete_user(
    user_id: str,
    admin_id: str = Depends(verify_admin),
) -> dict[str, str]:
    """
    Soft-delete user: set is_active=false di profiles.
    Tidak menghapus data dari Supabase Auth.

    Proteksi: Admin tidak bisa menghapus akunnya sendiri.

    Args:
        user_id: UUID user target untuk di-deactivate.
        admin_id: UUID admin dari JWT (injected by Depends).

    Returns:
        Dict konfirmasi deactivation.
    """
    if user_id == admin_id:
        raise HTTPException(
            status_code=400,
            detail="Tidak dapat menghapus akun diri sendiri.",
        )

    try:
        target = (
            supabase.table("profiles")
            .select("id")
            .eq("id", user_id)
            .execute()
        )
        if not target.data:
            raise HTTPException(status_code=404, detail="User tidak ditemukan.")

        supabase.table("profiles").update(
            {"is_active": False}
        ).eq("id", user_id).execute()

        logger.info(f"Admin {admin_id} deactivated user {user_id}")
        return {"message": f"User {user_id} berhasil dinonaktifkan."}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete user error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
