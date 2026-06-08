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
# PERSONA & MODEL TIER ROUTING DEFINITIONS
# ═══════════════════════════════════════════

from enum import Enum
from pydantic import BaseModel

class Persona(str, Enum):
    UMUM = "umum"
    PELAJAR = "pelajar"
    PENELITI = "peneliti"
    TENAGA_MEDIS = "tenaga_medis"

class ModelTier(str, Enum):
    FAST = "fast"
    THINKING = "thinking"

PERSONA_ALIASES = {
    "umum": Persona.UMUM,
    "general": Persona.UMUM,
    "pelajar": Persona.PELAJAR,
    "student": Persona.PELAJAR,
    "peneliti": Persona.PENELITI,
    "researcher": Persona.PENELITI,
    "tenaga_medis": Persona.TENAGA_MEDIS,
    "tenaga medis": Persona.TENAGA_MEDIS,
    "medical": Persona.TENAGA_MEDIS,
}

class ModelRoute(BaseModel):
    used_model: str
    model_tier: ModelTier
    requested_model: Optional[str] = None
    provider: str = "hf_router"
    fallback_used: bool = False

LEGACY_UNSUPPORTED_MODELS = {
    "google/gemma-2-9b-it",
    "google/gemma-2-27b-it",
    "Qwen/Qwen2.5-14B-Instruct",
}

def resolve_model(
    model_tier: Optional[ModelTier] = None,
    requested_model: Optional[str] = None,
) -> ModelRoute:
    """
    Resolve model based on model_tier and requested_model with legacy normalization.
    """
    fallback_used = False
    resolved_tier = ModelTier.FAST

    # 1. Normalize legacy models first
    normalized_model = requested_model
    if requested_model in LEGACY_UNSUPPORTED_MODELS:
        fallback_used = True
        if requested_model == "google/gemma-2-9b-it":
            normalized_model = settings.MODEL_FAST
            resolved_tier = ModelTier.FAST
        else:
            normalized_model = settings.MODEL_THINKING
            resolved_tier = ModelTier.THINKING
        logger.warning(
            f"Legacy unsupported model normalized: requested={requested_model} resolved={normalized_model}"
        )
    else:
        # Determine tier from model_tier or model ID
        if model_tier == ModelTier.THINKING or normalized_model == settings.MODEL_THINKING:
            resolved_tier = ModelTier.THINKING
        elif model_tier == ModelTier.FAST or normalized_model == settings.MODEL_FAST:
            resolved_tier = ModelTier.FAST
        else:
            resolved_tier = ModelTier.FAST

    # 2. Select final model ID
    if resolved_tier == ModelTier.THINKING:
        used_model = settings.MODEL_THINKING
    else:
        used_model = settings.MODEL_FAST

    return ModelRoute(
        used_model=used_model,
        model_tier=resolved_tier,
        requested_model=requested_model,
        provider="hf_router",
        fallback_used=fallback_used
    )

def resolve_model_for_role(role: str, model_choice: Optional[str] = None) -> str:
    """
    Wrapper for backward compatibility.
    """
    tier = ModelTier.FAST
    if model_choice == settings.MODEL_THINKING or role in ("tenaga_medis", "peneliti"):
        tier = ModelTier.THINKING

    route = resolve_model(tier, model_choice)
    return route.used_model


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
