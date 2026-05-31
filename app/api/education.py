"""
Education API - Endpoint untuk modul edukasi.

Fitur:
- Pencarian materi edukasi kimia/farmasi/biologi.
- Penjelasan materi edukasi dengan formatting LLM.

Menggunakan hybrid search (vector + graph) dan dilindungi
oleh JWT authentication.
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from app.agent.llm_formatter import generate_strict_response
from app.agent.retriever import retrieve_education_corpus
from app.core.dependencies import verify_user
from app.models.schemas import SearchRequest

logger = logging.getLogger(__name__)
router = APIRouter()

# Thread pool untuk blocking I/O (retrieval + LLM calls)
_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="education")


def _explain_sync(query: str, limit: int) -> dict[str, Any]:
    """
    Pipeline sinkron penjelasan edukasi: retrieval + LLM formatting.

    Args:
        query: Topik atau pertanyaan edukasi.
        limit: Jumlah hasil retrieval.

    Returns:
        Dict berisi query dan explanation text.
    """
    context = retrieve_education_corpus(query, limit)
    explanation = generate_strict_response(
        query=query,
        context=context,
        ai_mode="Pelajar",
        intent="edukasi",
    )
    return {
        "query": query,
        "explanation": explanation,
    }


@router.post("/search", summary="Cari materi edukasi")
async def search_education_material(
    req: SearchRequest,
    user_id: str = Depends(verify_user),
) -> dict[str, Any]:
    """
    Pencarian materi edukasi menggunakan hybrid search (vector + graph).

    Mengembalikan konteks mentah dari retriever tanpa LLM formatting.
    Cocok untuk rendering custom di frontend.

    Args:
        req: SearchRequest berisi query dan limit.
        user_id: UUID user dari JWT (injected by Depends).

    Returns:
        Dict berisi query, context, dan result_count.
    """
    try:
        loop = asyncio.get_event_loop()
        context = await loop.run_in_executor(
            _executor,
            retrieve_education_corpus,
            req.query,
            req.limit,
        )
        return {
            "query": req.query,
            "context": context,
            "result_count": req.limit,
        }
    except Exception as e:
        logger.error(f"Education search error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/explain", summary="Jelaskan materi edukasi dengan AI")
async def explain_education_topic(
    req: SearchRequest,
    user_id: str = Depends(verify_user),
) -> dict[str, Any]:
    """
    Mencari materi edukasi lalu di-format oleh LLM menjadi penjelasan terstruktur.

    Menggunakan persona 'Pelajar' secara default untuk gaya bahasa
    yang edukatif dan mudah dipahami.

    Args:
        req: SearchRequest berisi query dan limit.
        user_id: UUID user dari JWT (injected by Depends).

    Returns:
        Dict berisi query dan explanation text.
    """
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            _executor,
            _explain_sync,
            req.query,
            req.limit,
        )
        return result
    except Exception as e:
        logger.error(f"Education explain error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
