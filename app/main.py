"""
Enterprise GraphRAG Agentic AI - Main Application Entry Point.

Pharmaceutical & Herbal Encyclopedia Backend.
Menginisialisasi FastAPI app, middleware, lifespan events, dan router registration.
"""

import logging
from contextlib import asynccontextmanager
from typing import Any

import redis.asyncio as redis
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi_limiter import FastAPILimiter

from app.api import admin, auth, chat, education, recommendation, upload
from app.core.config import settings
from app.core.database import close_connections, verify_neo4j_connection
from app.core.minio_client import ensure_bucket_exists

# ═══════════════════════════════════════════
# LOGGING CONFIGURATION
# ═══════════════════════════════════════════
logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(name)s] - %(message)s",
    handlers=[
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════
# APPLICATION LIFESPAN (Startup & Shutdown)
# ═══════════════════════════════════════════
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Mengelola lifecycle aplikasi:

    Startup:
    - Init Redis connection untuk rate limiter.
    - Verify Neo4j graph database connectivity.
    - Ensure MinIO bucket exists untuk file uploads.

    Shutdown:
    - Close Redis connection.
    - Close Neo4j driver.
    """
    logger.info("=" * 60)
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info("=" * 60)

    redis_connection = None

    # ── Redis Rate Limiter ──
    try:
        redis_connection = redis.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=True,
        )
        await FastAPILimiter.init(redis_connection)
        logger.info("Redis Rate Limiter initialized successfully.")
    except Exception as e:
        logger.warning(f"Redis connection failed (rate limiting disabled): {e}")

    # ── Neo4j Connectivity Check ──
    try:
        verify_neo4j_connection()
    except Exception as e:
        logger.warning(f"Neo4j connectivity check failed: {e}")

    # ── MinIO Bucket ──
    try:
        ensure_bucket_exists()
    except Exception as e:
        logger.warning(f"MinIO bucket check skipped: {e}")

    logger.info("All services initialized. Server is ready.")
    logger.info("=" * 60)

    yield

    # ── Shutdown ──
    logger.info("Shutting down services...")
    if redis_connection:
        try:
            await redis_connection.close()
            logger.info("Redis connection closed.")
        except Exception as e:
            logger.error(f"Error closing Redis connection: {e}")
    close_connections()
    logger.info("All services closed. Goodbye.")


# ═══════════════════════════════════════════
# FASTAPI APPLICATION
# ═══════════════════════════════════════════
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "Backend API untuk Ensiklopedia Farmasi & Tanaman Obat Indonesia "
        "dengan Agentic AI, GraphRAG, dan Zero-Hallucination."
    ),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS Middleware ──
cors_origins: list[str] = [
    origin.strip() for origin in settings.CORS_ORIGINS.split(",")
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ═══════════════════════════════════════════
# ROUTER REGISTRATION
# ═══════════════════════════════════════════
app.include_router(
    auth.router,
    prefix="/api/auth",
    tags=["Autentikasi & Security"],
)
app.include_router(
    admin.router,
    prefix="/api/admin",
    tags=["Dashboard Admin"],
)
app.include_router(
    chat.router,
    prefix="/api/chat",
    tags=["Agent & Chat Management"],
)
app.include_router(
    recommendation.router,
    prefix="/api/medis",
    tags=["Modul Medis"],
)
app.include_router(
    education.router,
    prefix="/api/edukasi",
    tags=["Modul Edukasi"],
)
app.include_router(
    upload.router,
    prefix="/api/files",
    tags=["Multimodal OCR"],
)


# ═══════════════════════════════════════════
# HEALTH CHECK
# ═══════════════════════════════════════════
@app.get("/", tags=["System"])
async def health_check() -> dict[str, str]:
    """Health check endpoint untuk monitoring dasar."""
    return {
        "status": "ok",
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
    }


@app.get("/health", tags=["System"])
async def detailed_health_check() -> dict[str, Any]:
    """
    Detailed health check termasuk status koneksi ke semua services.

    Memeriksa:
    - Supabase (PostgreSQL + pgvector + Auth)
    - Neo4j (Graph Database)
    - Redis (Rate Limiting)
    - MinIO (Object Storage)

    Returns:
        Dict dengan status overall dan per-service.
    """
    health: dict[str, Any] = {
        "status": "ok",
        "services": {
            "supabase": "connected",
            "neo4j": "unknown",
            "redis": "unknown",
            "minio": "unknown",
        },
    }

    # Check Neo4j
    try:
        neo4j_ok = verify_neo4j_connection()
        health["services"]["neo4j"] = "connected" if neo4j_ok else "disconnected"
    except Exception:
        health["services"]["neo4j"] = "error"

    # Check Redis
    try:
        r = redis.from_url(settings.REDIS_URL)
        await r.ping()
        health["services"]["redis"] = "connected"
        await r.close()
    except Exception:
        health["services"]["redis"] = "disconnected"

    # Check MinIO
    try:
        from app.core.minio_client import minio_client as mc
        mc.list_buckets()
        health["services"]["minio"] = "connected"
    except Exception:
        health["services"]["minio"] = "disconnected"

    # Overall status
    if any(v != "connected" for v in health["services"].values()):
        health["status"] = "degraded"

    return health
