"""Runtime schema adapter for the actual herbal Neo4j graph."""

import logging
from functools import lru_cache
from typing import Any

from pydantic import BaseModel

from app.core.database import neo4j_driver

logger = logging.getLogger(__name__)


class HerbalGraphSchema(BaseModel):
    herb_label: str = "Herb"
    local_name_property: str = "commonName"
    scientific_name_property: str = "latinName"
    alias_property: str | None = "localNames"
    description_property: str | None = "macroscopicDesc"
    therapeutic_use_relationship: str | None = "USED_FOR"
    compound_relationship: str | None = "HAS_COMPOUND"
    compound_class_relationship: str | None = "HAS_COMPOUND_CLASS"
    contains_class_relationship: str | None = "CONTAINS_CLASS"
    toxicity_relationship: str | None = "HAS_TOXICITY"
    source_relationship: str | None = "VERIFIED_BY"
    preparation_relationship: str | None = None
    usage_rule_relationship: str | None = None
    contraindication_relationship: str | None = None
    interaction_relationship: str | None = None
    warning_relationship: str | None = None
    availability_relationship: str | None = None
    evidence_relationship: str | None = None
    available_herb_properties: list[str] = []
    available_relationships: list[str] = []
    indexes: list[dict[str, Any]] = []
    constraints: list[dict[str, Any]] = []


# ---------------------------------------------------------------------------
# Graph capability registry
# ---------------------------------------------------------------------------

class HerbalGraphCapabilities(BaseModel):
    """Tracks which graph relationships are actually present in Neo4j."""

    base_candidate_retrieval: bool = False
    therapeutic_use: bool = False
    compounds: bool = False
    toxicity: bool = False
    provenance: bool = False

    preparation: bool = False
    usage_rule: bool = False
    contraindication: bool = False
    interaction: bool = False
    side_effect: bool = False
    risk_group: bool = False
    availability: bool = False
    warning: bool = False


def build_graph_capabilities(relationships: set[str]) -> HerbalGraphCapabilities:
    """Build capability flags from the set of available relationship types."""
    caps = HerbalGraphCapabilities(
        base_candidate_retrieval="USED_FOR" in relationships,
        therapeutic_use="USED_FOR" in relationships,
        compounds="HAS_COMPOUND" in relationships or "HAS_COMPOUND_CLASS" in relationships,
        toxicity="HAS_TOXICITY" in relationships,
        provenance="VERIFIED_BY" in relationships,
        preparation="HAS_PREPARATION" in relationships,
        usage_rule="HAS_USAGE_RULE" in relationships,
        contraindication="HAS_CONTRAINDICATION" in relationships,
        interaction="HAS_INTERACTION" in relationships,
        side_effect="HAS_SIDE_EFFECT" in relationships,
        risk_group="HAS_RISK_GROUP" in relationships,
        availability="HAS_AVAILABILITY" in relationships,
        warning="HAS_WARNING" in relationships,
    )
    logger.info(
        "herbal_graph_capabilities_resolved "
        "base_candidate_retrieval=%s therapeutic_use=%s compounds=%s "
        "toxicity=%s provenance=%s preparation=%s usage_rule=%s "
        "contraindication=%s interaction=%s side_effect=%s "
        "risk_group=%s availability=%s warning=%s",
        caps.base_candidate_retrieval, caps.therapeutic_use, caps.compounds,
        caps.toxicity, caps.provenance, caps.preparation, caps.usage_rule,
        caps.contraindication, caps.interaction, caps.side_effect,
        caps.risk_group, caps.availability, caps.warning,
    )
    return caps


# ---------------------------------------------------------------------------
# Neo4j error classification
# ---------------------------------------------------------------------------

def classify_neo4j_error(exc: Exception) -> str:
    """Map Neo4j driver exceptions to specific herbal error codes.

    Returns a fine-grained error code instead of the generic
    HERBAL_GRAPH_UNAVAILABLE for every exception.
    """
    from pydantic import ValidationError as PydanticValidationError

    try:
        from neo4j.exceptions import (
            AuthError,
            ClientError,
            ServiceUnavailable,
            SessionExpired,
            TransientError,
        )
    except ImportError:
        # Fallback if neo4j exceptions cannot be imported
        return "HERBAL_GRAPH_UNAVAILABLE"

    if isinstance(exc, AuthError):
        return "HERBAL_GRAPH_AUTH_FAILED"

    if isinstance(exc, (ServiceUnavailable, SessionExpired)):
        return "HERBAL_GRAPH_CONNECTION_FAILED"

    if isinstance(exc, TransientError):
        return "HERBAL_GRAPH_TIMEOUT"

    if isinstance(exc, ClientError):
        return "HERBAL_GRAPH_QUERY_INVALID"

    if isinstance(exc, PydanticValidationError):
        return "HERBAL_GRAPH_RESULT_INVALID"

    # Check for timeout-like messages in generic exceptions
    msg = str(exc).lower()
    if "timeout" in msg or "timed out" in msg or "deadline" in msg:
        return "HERBAL_GRAPH_TIMEOUT"

    return "HERBAL_GRAPH_UNAVAILABLE"


def error_status_code(error_code: str) -> int:
    """Map herbal error codes to HTTP status codes."""
    mapping = {
        "HERBAL_GRAPH_AUTH_FAILED": 503,
        "HERBAL_GRAPH_CONNECTION_FAILED": 503,
        "HERBAL_GRAPH_TIMEOUT": 504,
        "HERBAL_GRAPH_QUERY_INVALID": 500,
        "HERBAL_GRAPH_RESULT_INVALID": 500,
        "HERBAL_GRAPH_SCHEMA_INCOMPLETE": 500,
        "HERBAL_GRAPH_NO_MATCH": 200,
        "HERBAL_GRAPH_ENRICHMENT_FAILED": 200,  # base still works
        "HERBAL_GRAPH_UNAVAILABLE": 503,
    }
    return mapping.get(error_code, 500)


def is_retryable_error(error_code: str) -> bool:
    """Determine if an error is worth retrying."""
    return error_code in {
        "HERBAL_GRAPH_CONNECTION_FAILED",
        "HERBAL_GRAPH_TIMEOUT",
        "HERBAL_GRAPH_UNAVAILABLE",
    }


def _records(query: str) -> list[dict[str, Any]]:
    records, _, _ = neo4j_driver.execute_query(query)
    return [record.data() for record in records]


def audit_neo4j_schema() -> dict[str, Any]:
    labels = _records("CALL db.labels() YIELD label RETURN collect(label) AS labels")
    rels = _records("CALL db.relationshipTypes() YIELD relationshipType RETURN collect(relationshipType) AS relationship_types")
    props = _records("CALL db.propertyKeys() YIELD propertyKey RETURN collect(propertyKey) AS property_keys")
    indexes = _records("SHOW INDEXES YIELD name, type, labelsOrTypes, properties RETURN collect({name:name,type:type,labelsOrTypes:labelsOrTypes,properties:properties}) AS indexes")
    constraints = _records("SHOW CONSTRAINTS YIELD name, type, labelsOrTypes, properties RETURN collect({name:name,type:type,labelsOrTypes:labelsOrTypes,properties:properties}) AS constraints")
    herb_property_groups = _records("MATCH (h:Herb) RETURN keys(h) AS property_keys, count(*) AS total ORDER BY total DESC")
    herb_samples = _records("MATCH (h:Herb) RETURN elementId(h) AS id, properties(h) AS properties, keys(h) AS keys LIMIT 20")
    outgoing = _records("MATCH (h:Herb)-[r]->(n) RETURN type(r) AS relationship, labels(n) AS target_labels, keys(r) AS relationship_properties, keys(n) AS target_properties, count(*) AS total ORDER BY total DESC")
    incoming = _records("MATCH (n)-[r]->(h:Herb) RETURN type(r) AS relationship, labels(n) AS source_labels, keys(r) AS relationship_properties, keys(n) AS source_properties, count(*) AS total ORDER BY total DESC")
    kencur = _records('''MATCH (h:Herb)
WHERE toLower(coalesce(h.commonName, "")) CONTAINS "kencur"
   OR toLower(coalesce(h.latinName, "")) CONTAINS "kaempferia galanga"
RETURN elementId(h) AS herb_id, h.commonName AS common_name, h.latinName AS scientific_name, properties(h) AS properties''')
    duplicates = _records('''MATCH (h:Herb)
WITH toLower(trim(coalesce(h.canonicalScientificName, h.latinName, ""))) AS normalized_scientific_name,
     collect(h) AS nodes, count(*) AS total
WHERE normalized_scientific_name <> "" AND total > 1
RETURN normalized_scientific_name,
       [node IN nodes | {id: elementId(node), commonName: node.commonName, latinName: node.latinName}] AS duplicate_nodes,
       total''')
    return {
        "labels": labels[0].get("labels", []) if labels else [],
        "relationship_types": rels[0].get("relationship_types", []) if rels else [],
        "property_keys": props[0].get("property_keys", []) if props else [],
        "indexes": indexes[0].get("indexes", []) if indexes else [],
        "constraints": constraints[0].get("constraints", []) if constraints else [],
        "herb_property_groups": herb_property_groups,
        "herb_samples": herb_samples,
        "outgoing_relationships": outgoing,
        "incoming_relationships": incoming,
        "kencur_nodes": kencur,
        "duplicate_scientific_names": duplicates,
    }


def _first_existing(candidates: list[str], available: set[str]) -> str | None:
    return next((item for item in candidates if item in available), None)


@lru_cache(maxsize=1)
def load_herbal_graph_schema() -> HerbalGraphSchema:
    audit = audit_neo4j_schema()
    properties = set(audit.get("property_keys", []))
    relationships = set(audit.get("relationship_types", []))
    return HerbalGraphSchema(
        local_name_property=_first_existing(["commonName", "name"], properties) or "commonName",
        scientific_name_property=_first_existing(["latinName", "scientificName"], properties) or "latinName",
        alias_property=_first_existing(["localNames", "aliases"], properties),
        description_property=_first_existing(["macroscopicDesc", "microscopicDesc", "simplisiaName"], properties),
        therapeutic_use_relationship=_first_existing(["USED_FOR", "TREATS", "HAS_THERAPEUTIC_USE"], relationships),
        compound_relationship=_first_existing(["HAS_COMPOUND"], relationships),
        compound_class_relationship=_first_existing(["HAS_COMPOUND_CLASS"], relationships),
        contains_class_relationship=_first_existing(["CONTAINS_CLASS"], relationships),
        toxicity_relationship=_first_existing(["HAS_TOXICITY"], relationships),
        source_relationship=_first_existing(["VERIFIED_BY"], relationships),
        preparation_relationship=_first_existing(["HAS_PREPARATION", "PREPARED_AS", "HAS_PREPARATION_METHOD"], relationships),
        usage_rule_relationship=_first_existing(["HAS_USAGE_RULE", "HAS_USAGE", "USED_AS"], relationships),
        contraindication_relationship=_first_existing(["HAS_CONTRAINDICATION", "CONTRAINDICATED_FOR"], relationships),
        interaction_relationship=_first_existing(["HAS_INTERACTION", "INTERACTS_WITH"], relationships),
        warning_relationship=_first_existing(["HAS_WARNING", "HAS_CAUTION"], relationships),
        availability_relationship=_first_existing(["HAS_AVAILABILITY", "AVAILABLE_AS"], relationships),
        evidence_relationship=_first_existing(["HAS_EVIDENCE", "SUPPORTED_BY"], relationships),
        available_herb_properties=sorted(properties),
        available_relationships=sorted(relationships),
        indexes=audit.get("indexes", []),
        constraints=audit.get("constraints", []),
    )


def clear_herbal_graph_schema_cache() -> None:
    load_herbal_graph_schema.cache_clear()


def graph_schema_log_fields(schema: HerbalGraphSchema) -> dict[str, Any]:
    return {
        "available_herb_properties": ",".join(schema.available_herb_properties),
        "available_relationships": ",".join(schema.available_relationships),
    }
