"""
Shared FastAPI Dependencies - JWT Authentication & RBAC Authorization.

Modul ini menyediakan reusable dependency injection untuk:
- Verifikasi JWT token via Authorization header.
- Pengecekan role 'admin' di tabel profiles (RBAC).

Digunakan oleh seluruh API router agar tidak duplikasi logik auth (DRY).
"""

import logging
from typing import Optional

from fastapi import Header, HTTPException

from app.core.database import supabase

logger = logging.getLogger(__name__)


async def verify_user(authorization: Optional[str] = Header(None)) -> str:
    """
    FastAPI Dependency: Verifikasi JWT token Supabase.

    Mengekstrak Bearer token dari header Authorization,
    memvalidasi via Supabase Auth, dan mengembalikan user_id.

    Args:
        authorization: Header Authorization berformat 'Bearer <token>'.

    Returns:
        user_id (str): UUID dari user yang terautentikasi.

    Raises:
        HTTPException 401: Jika token tidak ada, format salah, atau tidak valid.
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Token Authorization diperlukan. Format: Bearer <token>",
        )

    token = authorization.split(" ", maxsplit=1)[1]

    try:
        user_res = supabase.auth.get_user(token)
        if not user_res or not user_res.user:
            raise HTTPException(
                status_code=401,
                detail="Token tidak valid atau sudah kadaluarsa.",
            )
        return user_res.user.id
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Token verification failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=401,
            detail="Token tidak valid atau sudah kadaluarsa.",
        )


async def verify_admin(authorization: Optional[str] = Header(None)) -> str:
    """
    FastAPI Dependency: Verifikasi JWT token + cek role admin.

    Melakukan dua langkah verifikasi:
    1. Validasi JWT token via Supabase Auth.
    2. Cek kolom 'role' di tabel 'profiles' harus bernilai 'admin'.

    Args:
        authorization: Header Authorization berformat 'Bearer <token>'.

    Returns:
        user_id (str): UUID dari admin yang terautentikasi.

    Raises:
        HTTPException 401: Jika token tidak valid.
        HTTPException 403: Jika user bukan admin.
    """
    # Step 1: Verify token (reuse verify_user logic)
    user_id = await verify_user(authorization)

    # Step 2: Check admin role in profiles table
    try:
        profile = (
            supabase.table("profiles")
            .select("role")
            .eq("id", user_id)
            .execute()
        )
        if not profile.data or profile.data[0].get("role") != "admin":
            raise HTTPException(
                status_code=403,
                detail="Akses ditolak. Memerlukan hak akses Admin.",
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Admin role verification failed for {user_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=403,
            detail="Gagal memverifikasi role user.",
        )

    return user_id
