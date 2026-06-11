"""
Enterprise GraphRAG Agentic AI - Main Application Entry Point.

Pharmaceutical & Herbal Encyclopedia Backend.
Menginisialisasi FastAPI app, middleware, lifespan events, dan router registration.
"""

import logging
from contextlib import asynccontextmanager
from typing import Any

try:
    import redis.asyncio as redis
except Exception:
    redis = None
# Tambahkan 'Request' dari fastapi untuk membaca context metadata HTTP
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
try:
    from fastapi_limiter import FastAPILimiter
except Exception:
    class FastAPILimiter:  # type: ignore[no-redef]
        @staticmethod
        async def init(*args: Any, **kwargs: Any) -> None:
            return None

from app.api import admin, auth, chat, education, herbal_recommendations, recommendation, upload, quiz
from app.core.config import settings
from app.core.database import close_connections, verify_neo4j_connection
from app.core.minio_client import ensure_bucket_exists
from app.core.huggingface_vlm_client import HuggingFaceVlmClient
from app.agent.multimodal import set_vlm_client

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
logger.info(
    "Settings loaded: app_name=%s version=%s debug=%s",
    settings.APP_NAME,
    settings.APP_VERSION,
    settings.DEBUG,
)
logger.info("Settings source file: %s", __file__)


# ═══════════════════════════════════════════
# ADVANCED RATE LIMIT DEFENSIVE IDENTIFIER
# ═══════════════════════════════════════════
async def rate_limit_identifier(request: Request) -> str:
    """
    Fungsi pembuat key pembatas rate limit yang aman untuk rute publik maupun privat.
    Mencegah error NoneType pada endpoint /login dengan melakukan fallback ke IP.
    """
    # 1. Coba deteksi jika user sudah terautentikasi melalui state middleware
    user = getattr(request.state, "user", None)
    if user and hasattr(user, "id") and user.id:
        return f"user:{user.id}:{request.url.path}"
    
    # 2. DEFENSIVE FALLBACK: Jika rute publik/anonim (seperti /login), gunakan Alamat IP Client
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        client_ip = forwarded_for.split(",")[0].strip()
    else:
        client_ip = request.client.host if request.client else "unknown_ip"
        
    return f"ip:{client_ip}:{request.url.path}"


# ═══════════════════════════════════════════
# APPLICATION LIFESPAN (Startup & Shutdown)
# ═══════════════════════════════════════════
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Mengelola lifecycle aplikasi:

    Startup:
    - Init Redis connection dengan Custom Identifier Rate Limiter.
    - Verify Neo4j graph database connectivity.
    - Ensure MinIO bucket exists untuk file uploads.

    Shutdown:
    - Close Redis connection.
    - Close Neo4j driver.
    """
    logger.info("=" * 60)
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info("=" * 60)

    try:
        from app.agent.quiz_generator import log_startup
        log_startup()
    except Exception as e:
        logger.warning(f"Failed to trigger quiz generator startup logs: {e}")

    redis_connection = None
    vlm_client = HuggingFaceVlmClient(settings)
    app.state.hf_vlm_client = vlm_client
    set_vlm_client(vlm_client)

    # ── Redis Rate Limiter ──
    try:
        if redis is None:
            raise RuntimeError("redis package unavailable")
        redis_connection = redis.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=True,
        )
        # Registrasikan fungsi kustom identifier kita di sini agar Redis tidak menerima NoneType
        await FastAPILimiter.init(redis_connection, identifier=rate_limit_identifier)
        logger.info("Redis Rate Limiter initialized successfully with IP-Fallback Identifier.")
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
    try:
        await vlm_client.aclose()
        set_vlm_client(None)
        logger.info("Hugging Face VLM client closed.")
    except Exception as e:
        logger.error(f"Error closing Hugging Face VLM client: {e}")
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
app.include_router(herbal_recommendations.router)
app.include_router(herbal_recommendations.health_router)
app.include_router(
    education.router,
    prefix="/api/edukasi",
    tags=["Modul Edukasi"],
)
app.include_router(
    upload.router,
    prefix="/api/files",
    tags=["Multimodal Attachment"],
)
app.include_router(
    quiz.router,
    prefix="/api/quiz",
    tags=["Chemistry Quiz Engine"],
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


@app.get("/api/health/models", tags=["System"])
async def health_models_endpoint() -> dict[str, Any]:
    """
    Returns the list of available models and tiers.
    """
    return {
        "models": [
            {
                "tier": "fast",
                "label": "Fast Medium",
                "model_id": settings.MODEL_FAST,
                "available": True,
                "provider": "hf_router"
            },
            {
                "tier": "thinking",
                "label": "Thinking High",
                "model_id": settings.MODEL_THINKING,
                "available": True,
                "provider": "hf_router"
            }
        ]
    }


@app.get("/health", tags=["System"])
async def detailed_health_check() -> dict[str, Any]:
    """
    Detailed health check termasuk status koneksi ke semua services.
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
@app.get("/api/health/vlm", tags=["System"])
async def health_vlm_endpoint(request: Request) -> dict[str, Any]:
    """Healthcheck VLM remote tanpa mengunduh atau menjalankan model lokal."""
    client: HuggingFaceVlmClient = request.app.state.hf_vlm_client
    return await client.healthcheck()


@app.get("/api/health/multimodal", tags=["System"])
async def health_multimodal_endpoint(request: Request) -> dict[str, Any]:
    """Healthcheck attachment multimodal remote VLM, Neo4j, dan MinIO."""
    from app.agent.verification import get_neo4j_schema_map
    from app.core.minio_client import minio_client as mc

    neo4j_available = False
    schema_loaded = False
    try:
        neo4j_available = verify_neo4j_connection()
        schema = await get_neo4j_schema_map()
        schema_loaded = bool(schema.compound_labels or schema.herb_labels)
    except Exception:
        neo4j_available = False
        schema_loaded = False

    minio_available = False
    try:
        mc.bucket_exists(settings.MINIO_BUCKET)
        minio_available = True
    except Exception:
        minio_available = False

    client: HuggingFaceVlmClient = request.app.state.hf_vlm_client
    vlm = await client.healthcheck()
    return {
        "vlm": vlm,
        "neo4j": {"available": neo4j_available, "schema_loaded": schema_loaded},
        "storage": {"minio_available": minio_available},
    }
