"""Dedicated grounded herbal recommendation API."""

import logging
import uuid
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Response

from app.agent.herbal_recommendation_service import (
    analyze_herbal_complaint,
    get_cached,
    refresh_cached,
)
from app.core.config import settings
from app.core.dependencies import verify_user
from app.models.herbal_recommendation import (
    HerbalRecommendationError,
    HerbalRecommendationRequest,
    HerbalRecommendationResponse,
)
from app.core.database import neo4j_driver
from app.agent.herbal_graph_schema import load_herbal_graph_schema, build_graph_capabilities

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/herbal-recommendations",
    tags=["Herbal Recommendations"],
)

health_router = APIRouter(
    tags=["Herbal Recommendations Health"],
)


@router.get("/health")
async def herbal_recommendation_health() -> dict[str, str]:
    return {
        "status": "ok",
        "feature": "herbal_recommendations",
        "analyze_endpoint": "/api/herbal-recommendations/analyze",
    }


@health_router.get("/api/health/herbal-graph")
async def herbal_graph_health() -> dict[str, Any]:
    connected = False
    base_retrieval_ready = False
    capabilities_dict = {
        "therapeutic_use": False,
        "compounds": False,
        "toxicity": False,
        "preparation": False,
        "usage_rule": False,
        "interaction": False,
        "availability": False,
    }

    try:
        # Run lightweight query
        records, _, _ = neo4j_driver.execute_query("RETURN 1 AS ok")
        if records:
            connected = True
    except Exception as exc:
        logger.warning(f"Healthcheck Neo4j connection failed: {exc}")
        connected = False

    if connected:
        try:
            schema = load_herbal_graph_schema()
            relationships = set(schema.available_relationships)
            caps = build_graph_capabilities(relationships)
            base_retrieval_ready = caps.base_candidate_retrieval
            capabilities_dict = {
                "therapeutic_use": caps.therapeutic_use,
                "compounds": caps.compounds,
                "toxicity": caps.toxicity,
                "preparation": caps.preparation,
                "usage_rule": caps.usage_rule,
                "interaction": caps.interaction,
                "availability": caps.availability,
            }
        except Exception as exc:
            logger.warning(f"Healthcheck Neo4j schema loading failed: {exc}")
            base_retrieval_ready = False

    return {
        "status": "ok" if connected and base_retrieval_ready else "degraded",
        "connected": connected,
        "base_retrieval_ready": base_retrieval_ready,
        "capabilities": capabilities_dict,
    }


def _error(exc: HerbalRecommendationError) -> HTTPException:
    return HTTPException(
        status_code=exc.status_code,
        detail={
            "error": {
                "code": exc.code,
                "message": exc.message,
                "retryable": exc.retryable,
            }
        },
    )


@router.post("/analyze", response_model=HerbalRecommendationResponse)
async def analyze(
    req: HerbalRecommendationRequest,
    response: Response,
    user_id: str = Depends(verify_user),
    x_request_id: str | None = Header(default=None, alias="X-Request-ID"),
) -> HerbalRecommendationResponse:
    request_id = x_request_id or str(uuid.uuid4())
    response.headers["X-Request-ID"] = request_id
    logger.info(
        "herbal_recommendation_requested user_id=%s complaint_length=%s model_id=%s",
        user_id,
        len(req.complaint),
        settings.MODEL_THINKING,
    )
    try:
        return analyze_herbal_complaint(req, user_id=user_id, request_id=request_id)
    except HerbalRecommendationError as exc:
        response.headers["X-Request-ID"] = request_id
        raise _error(exc) from exc
    except Exception as exc:
        logger.exception(
            "herbal_recommendation_unexpected_failed request_id=%s error_class=%s error_message=%s",
            request_id,
            exc.__class__.__name__,
            str(exc)[:500],
        )
        raise _error(HerbalRecommendationError(
            "HERBAL_RECOMMENDATION_FAILED",
            "Rekomendasi gagal diproses.",
            status_code=500,
            retryable=False,
        )) from exc


@router.get("/{recommendation_id}", response_model=HerbalRecommendationResponse)
async def get_recommendation(
    recommendation_id: str,
    response: Response,
    user_id: str = Depends(verify_user),
    x_request_id: str | None = Header(default=None, alias="X-Request-ID"),
) -> HerbalRecommendationResponse:
    request_id = x_request_id or str(uuid.uuid4())
    response.headers["X-Request-ID"] = request_id
    cached = get_cached(recommendation_id)
    if cached is None:
        raise HTTPException(status_code=404, detail={"error": {"code": "HERBAL_RECOMMENDATION_NOT_FOUND", "message": "Hasil rekomendasi tidak ditemukan.", "retryable": False}})
    return cached


@router.post("/{recommendation_id}/refresh", response_model=HerbalRecommendationResponse)
async def refresh_recommendation(
    recommendation_id: str,
    response: Response,
    user_id: str = Depends(verify_user),
    x_request_id: str | None = Header(default=None, alias="X-Request-ID"),
) -> HerbalRecommendationResponse:
    request_id = x_request_id or str(uuid.uuid4())
    response.headers["X-Request-ID"] = request_id
    try:
        refreshed = refresh_cached(recommendation_id, user_id=user_id, request_id=request_id)
    except HerbalRecommendationError as exc:
        raise _error(exc) from exc
    if refreshed is None:
        raise HTTPException(status_code=404, detail={"error": {"code": "HERBAL_RECOMMENDATION_NOT_FOUND", "message": "Hasil rekomendasi tidak ditemukan.", "retryable": False}})
    return refreshed
