"""
Shared FastAPI Dependencies - JWT Authentication & RBAC Authorization.

Modul ini menyediakan reusable dependency injection untuk:
- Verifikasi JWT token via Authorization header.
- Pengecekan role 'admin' di tabel profiles (RBAC).
- Verifikasi user + role untuk model selection guardrail.

Digunakan oleh seluruh API router agar tidak duplikasi logik auth (DRY).
"""

import logging
from typing import Optional

from fastapi import Header, HTTPException, Request

from app.core.config import settings
from app.core.database import supabase

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════
# ROLE-BASED MODEL ALLOWED LIST
# ═══════════════════════════════════════════

# Maps Supabase profile role → list of allowed model setting values
ROLE_ALLOWED_MODELS: dict[str, list[str]] = {
    "tenaga_medis": [settings.MODEL_MEDIS_1, settings.MODEL_MEDIS_2],
    "peneliti": [settings.MODEL_MEDIS_1, settings.MODEL_MEDIS_2],
    "pelajar": [settings.MODEL_PELAJAR_1, settings.MODEL_PELAJAR_2],
    "umum": [settings.MODEL_UMUM],
    "user": [
        settings.MODEL_UMUM,
        settings.MODEL_PELAJAR_1,
        settings.MODEL_PELAJAR_2,
        settings.MODEL_MEDIS_1,
        settings.MODEL_MEDIS_2,
    ],
}

# Maps role → default model (first in allowed list)
ROLE_DEFAULT_MODEL: dict[str, str] = {
    "tenaga_medis": settings.MODEL_MEDIS_1,
    "peneliti": settings.MODEL_MEDIS_1,
    "pelajar": settings.MODEL_PELAJAR_1,
    "umum": settings.MODEL_UMUM,
    "user": settings.MODEL_UMUM,
}


def resolve_model_for_role(role: str, model_choice: Optional[str] = None) -> str:
    """
    Validasi dan resolusi model berdasarkan role user.

    Guardrail logic:
    - Jika model_choice kosong/None -> gunakan default model untuk role.
    - Jika model_choice ada di config -> gunakan model_choice (bypass role-gating).
    - Jika model_choice ada tapi tidak termasuk allowed list role -> fallback ke default.
    - Jika model_choice valid untuk role -> gunakan model_choice.

    Args:
        role: Role user dari tabel profiles (tenaga_medis/peneliti/pelajar/umum).
        model_choice: Model pilihan dari frontend (opsional).

    Returns:
        String model identifier yang sudah tervalidasi.
    """
    default_model = ROLE_DEFAULT_MODEL.get(role, settings.LLM_DEFAULT_MODEL)
    allowed_models = ROLE_ALLOWED_MODELS.get(role, [settings.LLM_DEFAULT_MODEL])

    if not model_choice or model_choice.strip() == "":
        logger.info(
            f"No model_choice provided for role '{role}', "
            f"using default: {default_model}"
        )
        return default_model

    model_choice_stripped = model_choice.strip()
    configured_models = {
        settings.LLM_DEFAULT_MODEL,
        settings.MODEL_MEDIS_1,
        settings.MODEL_MEDIS_2,
        settings.MODEL_PELAJAR_1,
        settings.MODEL_PELAJAR_2,
        settings.MODEL_UMUM,
        settings.VLM_MODEL,
    }

    if model_choice_stripped in configured_models:
        logger.info(
            f"Model '{model_choice_stripped}' is configured in settings. "
            f"Bypassing role-gating for role '{role}'."
        )
        return model_choice_stripped

    if model_choice in allowed_models:
        logger.info(
            f"Model '{model_choice}' authorized for role '{role}'."
        )
        return model_choice

    logger.warning(
        f"Model '{model_choice}' NOT authorized for role '{role}'. "
        f"Allowed: {allowed_models}. Falling back to default: {default_model}"
    )
    return default_model


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


async def verify_user_with_role(
    authorization: Optional[str] = Header(None),
) -> dict[str, str]:
    """
    FastAPI Dependency: Verifikasi JWT token + ambil role dari profiles.

    Mengembalikan dict berisi user_id dan role untuk digunakan
    oleh endpoint yang memerlukan role-based model selection.

    Args:
        authorization: Header Authorization berformat 'Bearer <token>'.

    Returns:
        Dict {"user_id": str, "role": str}.

    Raises:
        HTTPException 401: Jika token tidak valid.
    """
    user_id = await verify_user(authorization)

    # Ambil role dari tabel profiles
    try:
        profile = (
            supabase.table("profiles")
            .select("role")
            .eq("id", user_id)
            .execute()
        )
        role = "umum"  # Default fallback
        if profile.data and profile.data[0].get("role"):
            role = profile.data[0]["role"]
    except Exception as e:
        logger.warning(
            f"Failed to fetch role for user {user_id}: {e}. "
            "Defaulting to 'umum'."
        )
        role = "umum"

    return {"user_id": user_id, "role": role}


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


async def verify_pelajar(
    request: Request,
    authorization: Optional[str] = Header(None),
) -> str:
    """
    FastAPI Dependency: Verifikasi JWT token + cek role pelajar.

    Melakukan langkah verifikasi:
    1. Validasi JWT token via Supabase Auth.
    2. Cek kolom 'role' di tabel 'profiles' harus bernilai 'pelajar'.
    3. Simpan objek user di request.state.user.

    Args:
        request: FastAPI Request object.
        authorization: Header Authorization berformat 'Bearer <token>'.

    Returns:
        user_id (str): UUID dari pelajar yang terautentikasi.

    Raises:
        HTTPException 401: Jika token tidak valid.
        HTTPException 403: Jika user bukan pelajar.
    """
    user_id = await verify_user(authorization)

    try:
        profile = (
            supabase.table("profiles")
            .select("role")
            .eq("id", user_id)
            .execute()
        )
        if not profile.data:
            raise HTTPException(
                status_code=403,
                detail="Akses ditolak. Profil tidak ditemukan.",
            )
        role = profile.data[0].get("role", "umum")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Pelajar role verification failed for {user_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=403,
            detail="Gagal memverifikasi role user.",
        )

    # Simpan ke request.state.user
    class RequestUser:
        def __init__(self, uid: str, r: str):
            self.id = uid
            self.role = r

    request.state.user = RequestUser(user_id, role)

    # Strict check
    if request.state.user.role != "pelajar":
        raise HTTPException(
            status_code=403,
            detail="Akses ditolak. Fitur kuis hanya tersedia untuk akun Pelajar.",
        )

    return user_id
