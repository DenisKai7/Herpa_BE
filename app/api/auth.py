"""
Auth API - Registration, Login, Logout & Profile endpoints.

Keamanan:
- Login dilindungi oleh Redis Rate Limiter (5 req/60s per IP).
- Endpoint profil menggunakan JWT verification via shared dependency.
- Registration mengumpulkan data profil lengkap (instansi, provinsi, kota).
"""

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
try:
    from fastapi_limiter.depends import RateLimiter
except Exception:
    def RateLimiter(*args: Any, **kwargs: Any):  # type: ignore[no-redef]
        async def _noop() -> None:
            return None
        return _noop
try:
    from gotrue.errors import AuthApiError
except Exception:
    class AuthApiError(Exception):
        pass

from app.core.database import supabase
from app.core.dependencies import verify_user
from app.models.schemas import (
    AuthResponse,
    LoginRequest,
    MessageResponse,
    RegisterRequest,
)

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "/register",
    summary="Registrasi user baru",
    status_code=201,
)
async def register(req: RegisterRequest) -> AuthResponse | MessageResponse:
    """
    Mendaftarkan user baru dengan Supabase Auth lalu auto-login.

    Metadata profil (username, nama, instansi, provinsi, kota) disimpan
    di user_metadata. Supabase trigger `handle_new_user()` akan otomatis
    membuat row di tabel 'profiles'.

    Returns:
        AuthResponse (token + user) jika auto-login berhasil,
        atau MessageResponse jika email confirmation diperlukan.
    """
    try:
        res = supabase.auth.sign_up({
            "email": req.email,
            "password": req.password,
            "options": {
                "data": {
                    "username": req.username,
                    "nama": req.nama,
                    "instansi": req.instansi,
                    "provinsi": req.provinsi,
                    "kota": req.kota,
                }
            },
        })

        if res.user is None:
            raise HTTPException(
                status_code=400,
                detail="Registrasi gagal. Email mungkin sudah terdaftar.",
            )

        # Auto-login setelah register jika session tersedia
        if res.session:
            logger.info(f"User registered and logged in: {req.email}")
            return AuthResponse(
                token=res.session.access_token,
                user={
                    "id": res.user.id,
                    "email": res.user.email,
                    "username": res.user.user_metadata.get("username"),
                    "nama": res.user.user_metadata.get("nama"),
                    "role": res.user.user_metadata.get("role", "user"),
                },
            )

        # Session tidak tersedia = email confirmation aktif di Supabase
        logger.info(f"User registered (email confirmation required): {req.email}")
        return MessageResponse(
            message="Registrasi berhasil. Silakan cek email untuk verifikasi, lalu login.",
            data={"user_id": res.user.id},
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Registration error for {req.email}: {e}", exc_info=True)
        raise HTTPException(
            status_code=400,
            detail=f"Registrasi gagal: {str(e)}",
        )


@router.post(
    "/login",
    summary="Login user",
    response_model=AuthResponse,
    dependencies=[Depends(RateLimiter(times=5, seconds=60))],
)
async def login(req: LoginRequest) -> AuthResponse:
    """
    Login dengan email & password via Supabase Auth.

    Dilindungi rate limiter: maksimal 5 request per 60 detik per IP.
    Mengembalikan JWT access_token untuk dipakai di header Authorization.

    Args:
        req: Kredensial login (email + password).

    Returns:
        AuthResponse berisi token JWT dan data user dasar.
    """
    try:
        res = supabase.auth.sign_in_with_password({
            "email": req.email,
            "password": req.password,
        })

        if res.session is None:
            raise HTTPException(
                status_code=401,
                detail="Login gagal. Periksa email dan password Anda.",
            )

        logger.info(f"User logged in: {req.email}")
        return AuthResponse(
            token=res.session.access_token,
            user={
                "id": res.user.id,
                "email": res.user.email,
                "username": res.user.user_metadata.get("username"),
                "nama": res.user.user_metadata.get("nama"),
                "role": res.user.user_metadata.get("role", "user"),
            },
        )

    except HTTPException:
        raise
    except AuthApiError as e:
        logger.warning(f"Login failed for {req.email}: {e.message}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Email atau password yang Anda masukkan salah. Silakan periksa kembali.",
        )
    except Exception as e:
        logger.error(f"Login error for {req.email}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Terjadi kesalahan server saat memproses login. Silakan coba lagi nanti.",
        )


@router.post("/logout", summary="Logout user", response_model=MessageResponse)
async def logout(
    user_id: str = Depends(verify_user),
) -> MessageResponse:
    """
    Logout user. Supabase JWT bersifat stateless, sehingga endpoint ini
    berfungsi sebagai semantic endpoint. Client harus menghapus token di frontend.

    Args:
        user_id: UUID user dari JWT token (divalidasi oleh dependency).

    Returns:
        MessageResponse konfirmasi logout.
    """
    logger.info(f"User logged out: {user_id}")
    return MessageResponse(
        message="Logout berhasil. Silakan hapus token di client.",
    )


@router.get("/me", summary="Get current user profile")
async def get_current_user(
    user_id: str = Depends(verify_user),
) -> dict[str, Any]:
    """
    Mengambil profil lengkap user yang sedang login.

    Menggabungkan data dari Supabase Auth (email) dan tabel profiles
    (username, nama, role, instansi, dll).

    Args:
        user_id: UUID user dari JWT token (divalidasi oleh dependency).

    Returns:
        Dict berisi gabungan data auth + profile.
    """
    try:
        profile = (
            supabase.table("profiles")
            .select("*")
            .eq("id", user_id)
            .execute()
        )
        profile_data: dict[str, Any] = profile.data[0] if profile.data else {}

        return {
            "id": user_id,
            **profile_data,
        }

    except Exception as e:
        logger.error(f"Get user profile error for {user_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Gagal mengambil profil user.",
        )
