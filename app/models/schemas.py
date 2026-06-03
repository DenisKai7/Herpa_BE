"""
Pydantic Schemas - Validasi request/response untuk seluruh API endpoint.

Memastikan type safety dan dokumentasi otomatis di Swagger/OpenAPI.
Semua schema menggunakan Pydantic v2 dengan strict validation.
"""

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, EmailStr


# ═══════════════════════════════════════════
# AUTH SCHEMAS
# ═══════════════════════════════════════════

class RegisterRequest(BaseModel):
    """Schema registrasi user baru dengan data profil lengkap."""

    email: EmailStr = Field(..., description="Email address pengguna")
    password: str = Field(
        ...,
        min_length=8,
        max_length=128,
        description="Password minimal 8 karakter",
    )
    username: str = Field(
        ...,
        min_length=3,
        max_length=50,
        description="Username unik",
    )
    nama: str = Field(
        ...,
        min_length=2,
        max_length=100,
        description="Nama lengkap",
    )
    instansi: str = Field(
        ...,
        description="Institusi: Universitas, Rumah Sakit, dll.",
    )
    provinsi: str = Field(..., description="Provinsi domisili")
    kota: str = Field(..., description="Kota/Kabupaten domisili")


class LoginRequest(BaseModel):
    """Schema login dengan rate limiting."""

    email: EmailStr
    password: str = Field(..., min_length=1)


class AuthResponse(BaseModel):
    """Response sukses setelah login."""

    token: str
    user: dict[str, Any]


# ═══════════════════════════════════════════
# CHAT SCHEMAS
# ═══════════════════════════════════════════

class ChatRequest(BaseModel):
    """Request utama untuk mengirim pesan ke AI Agent."""

    message: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Pesan teks dari pengguna",
    )
    chat_id: Optional[str] = Field(
        None,
        description="Chat session ID. Null = buat baru.",
    )
    ai_mode: str = Field(
        default="Umum",
        description="Persona AI: Tenaga Medis, Peneliti, Pelajar, Umum",
    )
    file_context: Optional[str] = Field(
        None,
        description="Teks hasil OCR dari file upload",
    )
    file_url: Optional[str] = Field(
        None,
        description="URL file upload (MinIO)",
    )
    file_name: Optional[str] = Field(
        None,
        description="Nama file asli",
    )
    file_type: Optional[str] = Field(
        None,
        description="MIME type file",
    )


class ChatActionRequest(BaseModel):
    """Schema untuk aksi manajemen chat (rename, pin, share)."""

    title: Optional[str] = Field(
        None,
        max_length=100,
        description="Judul baru untuk chat",
    )
    is_pinned: Optional[bool] = Field(
        None,
        description="Status pinned chat",
    )
    is_public: Optional[bool] = Field(
        None,
        description="Status publik/shared chat",
    )


class ChatListItem(BaseModel):
    """Item dalam daftar chat sidebar user."""

    id: str
    title: str
    is_pinned: bool = False
    is_public: bool = False
    created_at: datetime
    updated_at: Optional[datetime] = None


class MessageItem(BaseModel):
    """Representasi satu pesan dalam chat."""

    id: str
    role: Literal["user", "ai"]
    content: str
    metadata: Optional[dict[str, Any]] = None
    created_at: datetime


class ChatResponse(BaseModel):
    """Response dari endpoint chat/message."""

    chat_id: str
    intent: str
    response: str
    quiz_data: Optional[dict[str, Any]] = None


# ═══════════════════════════════════════════
# ADMIN SCHEMAS
# ═══════════════════════════════════════════

class RoleUpdateRequest(BaseModel):
    """Request mengubah role user oleh admin."""

    target_user_id: str = Field(
        ...,
        description="UUID user yang akan diubah rolenya",
    )
    new_role: Literal["admin", "user"] = Field(
        ...,
        description="Role baru: admin atau user",
    )


class AnalyticsResponse(BaseModel):
    """Response dashboard analytics admin."""

    total_users: int
    total_chat_sessions: int
    total_messages: int = 0
    status: str = "Healthy"


class UserListItem(BaseModel):
    """Item user dalam daftar admin panel."""

    id: str
    email: Optional[str] = None
    username: Optional[str] = None
    nama: Optional[str] = None
    role: str = "user"
    instansi: Optional[str] = None
    provinsi: Optional[str] = None
    kota: Optional[str] = None
    created_at: Optional[datetime] = None


# ═══════════════════════════════════════════
# EDUCATION & RECOMMENDATION SCHEMAS
# ═══════════════════════════════════════════

class SearchRequest(BaseModel):
    """Request pencarian ensiklopedia/edukasi."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Kata kunci pencarian",
    )
    limit: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Jumlah hasil pencarian",
    )


class EncyclopediaEntry(BaseModel):
    """Satu entri hasil pencarian ensiklopedia."""

    id: str
    nama: str
    nama_latin: Optional[str] = None
    deskripsi: Optional[str] = None
    khasiat: Optional[str] = None
    kategori: Optional[str] = None
    similarity_score: Optional[float] = None


class RecommendationRequest(BaseModel):
    """Request rekomendasi tanaman obat berbasis gejala."""

    gejala: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="Deskripsi gejala pengguna",
    )
    limit: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Jumlah rekomendasi",
    )


# ═══════════════════════════════════════════
# FILE UPLOAD SCHEMAS
# ═══════════════════════════════════════════

class UploadResponse(BaseModel):
    """Response dari upload file (PDF/Image/TXT)."""

    filename: str
    url: str
    extracted_text: str


# ═══════════════════════════════════════════
# GENERIC RESPONSE
# ═══════════════════════════════════════════

class MessageResponse(BaseModel):
    """Generic success/info message response."""

    message: str
    data: Optional[dict[str, Any]] = None
