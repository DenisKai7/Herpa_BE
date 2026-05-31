"""
Recommendation API - Endpoint untuk modul medis.

Fitur:
- Rekomendasi tanaman obat berdasarkan gejala (content-based).
- Pencarian ensiklopedia tanaman/senyawa.
- Raw search untuk debugging dan custom UI rendering.

Semua endpoint menggunakan hybrid search (vector + graph) dan
dilindungi oleh JWT authentication.
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from app.agent.llm_formatter import generate_strict_response
from app.agent.retriever import content_based_recommendation, search_encyclopedia
from app.core.dependencies import verify_user
from app.models.schemas import RecommendationRequest, SearchRequest

logger = logging.getLogger(__name__)
router = APIRouter()

# Thread pool untuk blocking I/O (retrieval + LLM calls)
_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="recommendation")


def _recommend_sync(gejala: str, limit: int) -> dict[str, Any]:
    """
    Pipeline sinkron rekomendasi: retrieval + LLM formatting.

    Args:
        gejala: Deskripsi gejala dari pengguna.
        limit: Jumlah hasil rekomendasi.

    Returns:
        Dict berisi gejala dan recommendation text.
    """
    context = content_based_recommendation(gejala, limit)
    recommendation = generate_strict_response(
        query=gejala,
        context=context,
        ai_mode="Umum",
        intent="konsultasi",
    )
    return {
        "gejala": gejala,
        "recommendation": recommendation,
    }


def _search_encyclopedia_sync(query: str, limit: int) -> dict[str, Any]:
    """
    Pipeline sinkron pencarian ensiklopedia: retrieval + LLM formatting.

    Args:
        query: Kata kunci pencarian.
        limit: Jumlah hasil pencarian.

    Returns:
        Dict berisi query dan formatted response.
    """
    context = search_encyclopedia(query, limit)
    formatted = generate_strict_response(
        query=query,
        context=context,
        ai_mode="Umum",
        intent="ensiklopedia",
    )
    return {
        "query": query,
        "response": formatted,
    }


@router.post("/recommend", summary="Rekomendasi tanaman obat berdasarkan gejala")
async def get_recommendation(
    req: RecommendationRequest,
    user_id: str = Depends(verify_user),
) -> dict[str, Any]:
    """
    Content-based recommendation: cari tanaman obat berdasarkan gejala/keluhan.

    Menggunakan hybrid search (vector similarity + graph traversal)
    kemudian di-format oleh LLM dengan zero-hallucination constraint.

    Args:
        req: RecommendationRequest berisi gejala dan limit.
        user_id: UUID user dari JWT (injected by Depends).

    Returns:
        Dict berisi gejala dan recommendation text.
    """
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            _executor,
            _recommend_sync,
            req.gejala,
            req.limit,
        )
        return result
    except Exception as e:
        logger.error(f"Recommendation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search", summary="Pencarian ensiklopedia tanaman/senyawa")
async def search_medical_encyclopedia(
    req: SearchRequest,
    user_id: str = Depends(verify_user),
) -> dict[str, Any]:
    """
    Pencarian ensiklopedia menggunakan hybrid search.

    Mengembalikan formatted response dari LLM berdasarkan konteks
    yang diperoleh dari vector search + graph traversal.

    Args:
        req: SearchRequest berisi query dan limit.
        user_id: UUID user dari JWT (injected by Depends).

    Returns:
        Dict berisi query dan response text.
    """
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            _executor,
            _search_encyclopedia_sync,
            req.query,
            req.limit,
        )
        return result
    except Exception as e:
        logger.error(f"Encyclopedia search error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/raw-search", summary="Pencarian ensiklopedia (raw context)")
async def search_raw(
    req: SearchRequest,
    user_id: str = Depends(verify_user),
) -> dict[str, Any]:
    """
    Pencarian ensiklopedia yang mengembalikan konteks mentah tanpa LLM formatting.

    Berguna untuk debugging atau custom UI rendering di frontend.

    Args:
        req: SearchRequest berisi query dan limit.
        user_id: UUID user dari JWT (injected by Depends).

    Returns:
        Dict berisi query dan raw_context string.
    """
    try:
        loop = asyncio.get_event_loop()
        context = await loop.run_in_executor(
            _executor,
            search_encyclopedia,
            req.query,
            req.limit,
        )
        return {
            "query": req.query,
            "raw_context": context,
        }
    except Exception as e:
        logger.error(f"Raw search error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
