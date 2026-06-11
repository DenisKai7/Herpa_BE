"""Neo4j retrieval adapter for herbal recommendations."""

import logging
import re
import time
from typing import Any

from neo4j import Query
from pydantic import ValidationError

from app.agent.herbal_graph_schema import (
    audit_neo4j_schema as _audit_neo4j_schema,
    build_graph_capabilities,
    classify_neo4j_error,
    error_status_code,
    is_retryable_error,
    load_herbal_graph_schema,
)
from app.core.config import settings
from app.core.database import neo4j_driver
from app.models.herbal_recommendation import HerbalRecommendationError

logger = logging.getLogger(__name__)


class HerbalGraphQueryError(HerbalRecommendationError):
    pass


class HerbalGraphResultError(HerbalRecommendationError):
    pass


class HerbalGraphEnrichmentError(HerbalRecommendationError):
    pass


def audit_neo4j_schema() -> dict[str, Any]:
    try:
        return _audit_neo4j_schema()
    except Exception as exc:
        code = classify_neo4j_error(exc)
        logger.exception(
            "herbal_graph_schema_audit_failed error_class=%s error_message=%s",
            exc.__class__.__name__,
            str(exc)[:500],
        )
        raise HerbalRecommendationError(
            code,
            "Layanan data herbal sedang tidak tersedia." if code.endswith("CONNECTION_FAILED") or code.endswith("UNAVAILABLE") else "Schema knowledge graph herbal belum valid.",
            status_code=error_status_code(code),
            retryable=is_retryable_error(code),
        ) from exc


def _terms(symptoms: list[str]) -> list[str]:
    values: list[str] = []
    for symptom in symptoms:
        text = str(symptom).lower().strip()
        if text:
            values.append(text)
            values.extend(re.findall(r"\b[a-zA-ZÀ-ÿ0-9]{4,}\b", text))
    return list(dict.fromkeys(values))[:40]


def _clean_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    text = str(value).strip()
    return [text] if text else []


def _query_error_message(code: str) -> str:
    if code == "HERBAL_GRAPH_TIMEOUT":
        return "Query knowledge graph herbal melebihi batas waktu."
    if code == "HERBAL_GRAPH_QUERY_INVALID":
        return "Query knowledge graph herbal perlu diperbaiki."
    if code == "HERBAL_GRAPH_RESULT_INVALID":
        return "Format hasil knowledge graph herbal tidak valid."
    if code in {"HERBAL_GRAPH_CONNECTION_FAILED", "HERBAL_GRAPH_AUTH_FAILED", "HERBAL_GRAPH_UNAVAILABLE"}:
        return "Layanan data herbal sedang tidak dapat dihubungi."
    return "Retrieval knowledge graph herbal gagal."


def _run_query(
    query_text: str,
    parameters: dict[str, Any],
    *,
    query_name: str,
    query_stage: str,
    timeout_seconds: int,
    request_id: str | None = None,
) -> list[dict[str, Any]]:
    started = time.perf_counter()
    record_count = 0
    attempts = settings.HERBAL_GRAPH_MAX_RETRIES + 1
    last_exc: Exception | None = None

    for attempt in range(1, attempts + 1):
        try:
            records, _, _ = neo4j_driver.execute_query(
                Query(query_text, timeout=timeout_seconds),
                parameters_=parameters,
            )
            rows: list[dict[str, Any]] = []
            for record in records:
                rows.append(record.data())
                record_count += 1
            return rows
        except Exception as exc:
            last_exc = exc
            code = classify_neo4j_error(exc)
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            logger.exception(
                "herbal_graph_retrieval_failed request_id=%s query_name=%s query_stage=%s "
                "error_class=%s error_message=%s elapsed_ms=%s timeout_seconds=%s "
                "record_count_before_failure=%s attempt=%s error_code=%s",
                request_id or "-",
                query_name,
                query_stage,
                exc.__class__.__name__,
                str(exc)[:500],
                elapsed_ms,
                timeout_seconds,
                record_count,
                attempt,
                code,
            )
            if attempt >= attempts or not is_retryable_error(code):
                raise HerbalGraphQueryError(
                    code,
                    _query_error_message(code),
                    status_code=error_status_code(code),
                    retryable=is_retryable_error(code),
                ) from exc
    raise last_exc or RuntimeError("unreachable graph query retry state")


BASE_CANDIDATE_QUERY = """
MATCH (h:Herb)-[:USED_FOR]->(use)
WHERE any(term IN $terms WHERE
    toLower(coalesce(use.name, "")) CONTAINS term OR
    toLower(coalesce(h.commonName, "")) CONTAINS term OR
    toLower(coalesce(h.latinName, "")) CONTAINS term OR
    toLower(coalesce(h.canonicalScientificName, "")) CONTAINS term OR
    any(n IN coalesce(h.localNames, []) WHERE toLower(toString(n)) CONTAINS term)
)
WITH DISTINCT h, collect(DISTINCT use.name) AS traditional_uses
CALL {
    WITH h
    OPTIONAL MATCH (h)-[:HAS_COMPOUND]->(compound)
    RETURN collect(DISTINCT compound.name) AS compounds
}
CALL {
    WITH h
    OPTIONAL MATCH (h)-[:HAS_COMPOUND_CLASS]->(compoundClass)
    RETURN collect(DISTINCT compoundClass.name) AS compound_classes_a
}
CALL {
    WITH h
    OPTIONAL MATCH (h)-[:CONTAINS_CLASS]->(containsClass)
    RETURN collect(DISTINCT containsClass.name) AS compound_classes_b
}
CALL {
    WITH h
    OPTIONAL MATCH (h)-[toxRel:HAS_TOXICITY]->(tox)
    RETURN collect(DISTINCT coalesce(tox.name, toxRel.oecdCategory, toxRel.ld50)) AS toxicity
}
CALL {
    WITH h
    OPTIONAL MATCH (h)-[:VERIFIED_BY]->(source)
    RETURN collect(DISTINCT source.id) AS source_ids,
           [s IN collect(DISTINCT source) WHERE s IS NOT NULL |
            {id:s.id, title:s.title, publisher:s.publisher, year:s.year,
             qualityGrade:s.qualityGrade, url:s.url, accessDate:s.accessDate, active:s.active}] AS sources
}
WITH h, traditional_uses, compounds, compound_classes_a, compound_classes_b, toxicity, source_ids, sources,
     [term IN $terms WHERE
        toLower(coalesce(h.commonName, "")) CONTAINS term OR
        toLower(coalesce(h.latinName, "")) CONTAINS term OR
        toLower(coalesce(h.canonicalScientificName, "")) CONTAINS term OR
        any(u IN traditional_uses WHERE toLower(toString(u)) CONTAINS term)
     ] AS matched_terms
RETURN
    elementId(h) AS herb_id,
    h.id AS database_id,
    h.commonName AS local_name,
    coalesce(h.canonicalScientificName, h.latinName) AS scientific_name,
    coalesce(h.localNames, []) AS aliases,
    matched_terms AS matched_symptoms,
    traditional_uses AS traditional_uses,
    compounds + compound_classes_a + compound_classes_b AS supported_activities,
    coalesce(h.activeCompounds, compounds) AS active_compounds,
    CASE WHEN size(compounds) + size(compound_classes_a) + size(compound_classes_b) > 0 THEN 'phytochemical_screening'
         WHEN size(traditional_uses) > 0 THEN 'traditional'
         ELSE 'data_not_available'
    END AS evidence_level,
    null AS availability,
    null AS availability_score,
    null AS availability_reason,
    toxicity AS toxicity,
    [] AS preparation_methods,
    [] AS usage_rules,
    [] AS contraindications,
    [] AS interactions,
    [] AS side_effects,
    [] AS risk_groups,
    [] AS warnings,
    [] AS stop_use_signs,
    source_ids AS source_ids,
    [elementId(h)] AS graph_node_ids,
    coalesce(h.dataVersion, 'herbal-recommendation-v1') AS data_version,
    toString(coalesce(h.lastVerifiedAt, datetime())) AS verified_at,
    sources AS sources
ORDER BY size(matched_terms) DESC, local_name ASC
LIMIT $limit
"""


def _enrichment_query_parts(capabilities: Any) -> list[tuple[str, str]]:
    parts: list[tuple[str, str]] = []
    if capabilities.preparation:
        parts.append(("preparation", """
MATCH (h:Herb) WHERE elementId(h) IN $herb_ids
OPTIONAL MATCH (h)-[:HAS_PREPARATION]->(prep)
RETURN elementId(h) AS herb_id, [p IN collect(DISTINCT prep) WHERE p IS NOT NULL | properties(p)] AS preparation_methods
"""))
    if capabilities.usage_rule:
        parts.append(("usage_rule", """
MATCH (h:Herb) WHERE elementId(h) IN $herb_ids
OPTIONAL MATCH (h)-[:HAS_USAGE_RULE]->(usage)
RETURN elementId(h) AS herb_id, [u IN collect(DISTINCT usage) WHERE u IS NOT NULL | properties(u)] AS usage_rules
"""))
    if capabilities.availability:
        parts.append(("availability", """
MATCH (h:Herb) WHERE elementId(h) IN $herb_ids
OPTIONAL MATCH (h)-[:HAS_AVAILABILITY]->(avail)
RETURN elementId(h) AS herb_id,
       CASE WHEN avail IS NULL THEN null ELSE avail.category END AS availability,
       CASE WHEN avail IS NULL THEN null ELSE avail.score END AS availability_score,
       CASE WHEN avail IS NULL THEN null ELSE avail.reason END AS availability_reason
"""))
    for field, rel in [
        ("contraindications", "HAS_CONTRAINDICATION"),
        ("interactions", "HAS_INTERACTION"),
        ("side_effects", "HAS_SIDE_EFFECT"),
        ("risk_groups", "HAS_RISK_GROUP"),
    ]:
        if getattr(capabilities, field[:-1] if field.endswith("s") else field, False) or (field == "side_effects" and capabilities.side_effect) or (field == "risk_groups" and capabilities.risk_group):
            parts.append((field, f"""
MATCH (h:Herb) WHERE elementId(h) IN $herb_ids
OPTIONAL MATCH (h)-[:{rel}]->(node)
RETURN elementId(h) AS herb_id, [n IN collect(DISTINCT node) WHERE n IS NOT NULL | properties(n)] AS {field}
"""))
    return parts


def retrieve_base_candidates(symptoms: list[str], max_results: int, *, request_id: str | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    schema = load_herbal_graph_schema()
    relationships = set(schema.available_relationships)
    capabilities = build_graph_capabilities(relationships)
    terms = _terms(symptoms)

    if not capabilities.base_candidate_retrieval:
        logger.warning("herbal_graph_schema_incomplete request_id=%s missing_relationships=USED_FOR", request_id or "-")
        return [], {
            "symptom_nodes": len(terms),
            "graph_records": 0,
            "candidate_count_raw": 0,
            "missing_relationships": ["USED_FOR"],
            "knowledge_graph_version": "herbal-recommendation-v1",
            "capabilities": capabilities.model_dump(),
        }
    if not terms:
        return [], {"symptom_nodes": 0, "graph_records": 0, "candidate_count_raw": 0, "capabilities": capabilities.model_dump()}

    started = time.perf_counter()
    rows = _run_query(
        BASE_CANDIDATE_QUERY,
        {"terms": terms, "limit": max_results},
        query_name="retrieve_base_candidates",
        query_stage="execute",
        timeout_seconds=settings.HERBAL_GRAPH_BASE_QUERY_TIMEOUT_SECONDS,
        request_id=request_id,
    )
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    candidates: list[dict[str, Any]] = []
    try:
        for row in rows:
            row["aliases"] = _clean_list(row.get("aliases"))
            for key in ["matched_symptoms", "traditional_uses", "supported_activities", "active_compounds", "toxicity"]:
                row[key] = _clean_list(row.get(key))
            row["graph_available_relationships"] = sorted(relationships)
            row["graph_missing_relationships"] = sorted(
                rel for rel in [
                    "HAS_PREPARATION", "HAS_USAGE_RULE", "HAS_AVAILABILITY",
                    "HAS_CONTRAINDICATION", "HAS_INTERACTION", "HAS_SIDE_EFFECT", "HAS_RISK_GROUP",
                ] if rel not in relationships
            )
            candidates.append(row)
    except (ValidationError, TypeError, ValueError) as exc:
        logger.exception(
            "herbal_graph_result_validation_failed request_id=%s query_name=retrieve_base_candidates "
            "error_class=%s error_message=%s record_count_before_failure=%s elapsed_ms=%s",
            request_id or "-", exc.__class__.__name__, str(exc)[:500], len(candidates), elapsed_ms,
        )
        raise HerbalGraphResultError(
            "HERBAL_GRAPH_RESULT_INVALID",
            _query_error_message("HERBAL_GRAPH_RESULT_INVALID"),
            status_code=500,
            retryable=False,
        ) from exc

    logger.info(
        "herbal_graph_base_retrieval_completed request_id=%s candidate_count=%s elapsed_ms=%s",
        request_id or "-", len(candidates), elapsed_ms,
    )
    return candidates, {
        "symptom_nodes": len(terms),
        "graph_records": len(candidates),
        "candidate_count_raw": len(candidates),
        "knowledge_graph_version": "herbal-recommendation-v1",
        "capabilities": capabilities.model_dump(),
        "missing_relationships": [k for k, v in capabilities.model_dump().items() if v is False],
    }


def enrich_candidates(candidates: list[dict[str, Any]], *, request_id: str | None = None) -> list[dict[str, Any]]:
    if not candidates:
        return candidates
    schema = load_herbal_graph_schema()
    capabilities = build_graph_capabilities(set(schema.available_relationships))
    skipped = [name for name, enabled in capabilities.model_dump().items() if name not in {"base_candidate_retrieval", "therapeutic_use", "compounds", "toxicity", "provenance"} and not enabled]
    if skipped:
        logger.info(
            "herbal_graph_enrichment_skipped request_id=%s reason=capability_not_available fields=%s",
            request_id or "-", ",".join(skipped),
        )

    herb_ids = [str(c.get("herb_id")) for c in candidates if c.get("herb_id")]
    by_id = {str(c.get("herb_id")): c for c in candidates if c.get("herb_id")}
    for field_name, query_text in _enrichment_query_parts(capabilities):
        try:
            rows = _run_query(
                query_text,
                {"herb_ids": herb_ids},
                query_name=f"enrich_{field_name}",
                query_stage="execute",
                timeout_seconds=settings.HERBAL_GRAPH_ENRICHMENT_TIMEOUT_SECONDS,
                request_id=request_id,
            )
            for row in rows:
                item = by_id.get(str(row.get("herb_id")))
                if item:
                    item.update({k: v for k, v in row.items() if k != "herb_id"})
        except HerbalRecommendationError as exc:
            logger.warning(
                "herbal_graph_enrichment_failed request_id=%s query_name=enrich_%s code=%s message=%s",
                request_id or "-", field_name, exc.code, exc.message,
            )
            raise HerbalGraphEnrichmentError(
                "HERBAL_GRAPH_ENRICHMENT_FAILED",
                "Sebagian data pengayaan knowledge graph tidak tersedia.",
                status_code=200,
                retryable=False,
            ) from exc
    return candidates


def retrieve_graph_verified_herbal_candidates(
    symptoms: list[str],
    max_results: int,
    request_id: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    base_candidates, graph_meta = retrieve_base_candidates(symptoms, max_results, request_id=request_id)
    if not base_candidates:
        return base_candidates, graph_meta
    try:
        candidates = enrich_candidates(base_candidates, request_id=request_id)
        graph_meta["partial_enrichment"] = False
    except HerbalGraphEnrichmentError:
        candidates = base_candidates
        graph_meta["partial_enrichment"] = True
    fully_verified = sum(1 for c in candidates if not c.get("graph_missing_relationships"))
    graph_meta.update({
        "fully_verified_count": fully_verified,
        "present_relationships": [r for r in load_herbal_graph_schema().available_relationships],
    })
    return candidates, graph_meta


def retrieve_herbal_candidates(symptoms: list[str], max_results: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    return retrieve_graph_verified_herbal_candidates(symptoms, max_results)
