"""
Neo4j Verification Module - Verifikasi hasil OCR gambar & dokumen menggunakan Graph Database.
"""

import asyncio
import logging
import re
from typing import Any, Optional

from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.database import neo4j_driver
from app.agent.multimodal import OcrExtractionResult

logger = logging.getLogger(__name__)

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# NEO4J SCHEMA ADAPTER
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

class Neo4jSchemaMap(BaseModel):
    herb_labels: list[str] = ["Herb", "Plant", "Tanaman"]
    compound_labels: list[str] = ["Compound", "Chemical", "Molecule", "Senyawa"]
    herb_name_properties: list[str] = ["commonName", "name", "nama", "nama_populer"]
    compound_name_properties: list[str] = ["name", "nama", "compoundName"]
    scientific_name_properties: list[str] = ["latinName", "scientificName", "nama_latin"]
    formula_properties: list[str] = ["formula", "molecularFormula", "rumus_kimia", "molecular_formula"]
    compound_relationships: list[str] = ["HAS_COMPOUND", "CONTAINS", "MENGANDUNG"]


_schema_map_cache: Optional[Neo4jSchemaMap] = None
_schema_map_lock = asyncio.Lock()


async def get_neo4j_schema_map() -> Neo4jSchemaMap:
    """Introspeksi schema Neo4j secara dinamis dan cache hasilnya."""
    global _schema_map_cache
    if _schema_map_cache is not None:
        return _schema_map_cache

    async with _schema_map_lock:
        if _schema_map_cache is not None:
            return _schema_map_cache

        try:
            loop = asyncio.get_event_loop()

            def _query_schema():
                with neo4j_driver.session() as session:
                    labels = [r["label"] for r in session.run("CALL db.labels()")]
                    rels = [r["relationshipType"] for r in session.run("CALL db.relationshipTypes()")]
                    props = [r["propertyKey"] for r in session.run("CALL db.propertyKeys()")]
                    return labels, rels, props

            labels, rels, props = await loop.run_in_executor(None, _query_schema)
            logger.info(f"Neo4j Introspection: labels={labels}, rels={rels}, props={props}")

            # Map labels
            herb_labels = [l for l in ["Herb", "Plant", "Tanaman", "Herbal"] if l in labels] or ["Herb"]
            compound_labels = [l for l in ["Compound", "Chemical", "Molecule", "Senyawa"] if l in labels] or ["Compound"]

            # Map properties
            herb_name_properties = [p for p in ["commonName", "name", "nama", "nama_populer"] if p in props] or ["commonName"]
            compound_name_properties = [p for p in ["name", "nama", "compoundName"] if p in props] or ["name"]
            scientific_name_properties = [p for p in ["latinName", "scientificName", "nama_latin"] if p in props] or ["latinName"]
            formula_properties = [p for p in ["formula", "molecularFormula", "rumus_kimia", "molecular_formula"] if p in props] or ["formula"]

            # Map relationships
            compound_relationships = [r for r in ["HAS_COMPOUND", "CONTAINS", "MENGANDUNG", "FOUND_IN"] if r in rels] or ["HAS_COMPOUND"]

            _schema_map_cache = Neo4jSchemaMap(
                herb_labels=herb_labels,
                compound_labels=compound_labels,
                herb_name_properties=herb_name_properties,
                compound_name_properties=compound_name_properties,
                scientific_name_properties=scientific_name_properties,
                formula_properties=formula_properties,
                compound_relationships=compound_relationships,
            )
            logger.info(f"Successfully constructed Neo4j Schema Map: {_schema_map_cache}")

        except Exception as e:
            logger.warning(f"Failed to introspect Neo4j schema, using defaults: {e}")
            _schema_map_cache = Neo4jSchemaMap()

        return _schema_map_cache

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# VERIFICATION SCHEMAS & LOGIC
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

class Neo4jCandidate(BaseModel):
    entity_type: str
    name: str
    scientific_name: Optional[str] = None
    formula: Optional[str] = None

    related_herbs: list[str] = Field(default_factory=list)
    matched_evidence: list[str] = Field(default_factory=list)

    score: float
    source: str = "neo4j"


class Neo4jVerificationResult(BaseModel):
    success: bool
    verification_status: str  # verified | partially_verified | ambiguous | insufficient_evidence | failed
    candidates: list[Neo4jCandidate] = Field(default_factory=list)
    confidence: float = 0.0
    warnings: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


async def verify_attachment_with_neo4j(
    extraction: OcrExtractionResult,
    user_query: str,
) -> Neo4jVerificationResult:
    """
    Verifikasi hasil ekstraksi OCR menggunakan Neo4j database.
    Mencocokkan entitas senyawa, formula, dan mengalkulasi score kebenaran.
    """
    if not settings.NEO4J_ATTACHMENT_VERIFICATION:
        return Neo4jVerificationResult(
            success=True,
            verification_status="not_applicable",
            warnings=["verification_disabled_by_config"],
        )

    if not extraction.success or not extraction.raw_text.strip():
        return Neo4jVerificationResult(
            success=False,
            verification_status="insufficient_evidence",
            warnings=["empty_ocr_extraction"],
        )

    try:
        schema_map = await get_neo4j_schema_map()
        loop = asyncio.get_event_loop()

        if not schema_map.compound_labels or not schema_map.compound_name_properties:
            return Neo4jVerificationResult(
                success=False,
                verification_status="failed",
                warnings=["neo4j_schema_missing_compound_mapping"],
                limitations=["Schema Neo4j tidak memiliki mapping label/properti senyawa yang dapat diverifikasi."],
            )
        # Gather search terms: chemical terms & molecular formulas from OCR
        search_terms = [t.lower().strip() for t in extraction.chemical_terms]
        search_formulas = [f.lower().strip() for f in extraction.molecular_formulas]

        # Extract words from text/query as fallback
        text_words = re.findall(r'\b\w{4,}\b', extraction.raw_text.lower())
        query_words = re.findall(r'\b\w{4,}\b', user_query.lower()) if user_query else []

        all_words = list(set(search_terms + text_words + query_words))

        # We construct the query parameters
        # Use first label/property mapping from the adapter
        compound_label = schema_map.compound_labels[0]
        herb_label = schema_map.herb_labels[0]
        rel_type = schema_map.compound_relationships[0]

        name_prop = schema_map.compound_name_properties[0]
        formula_prop = schema_map.formula_properties[0]

        herb_name_prop = schema_map.herb_name_properties[0]
        scientific_name_prop = schema_map.scientific_name_properties[0]

        # Cypher Query to fetch potential compound candidates
        cypher = f"""
        MATCH (c:{compound_label})
        WHERE toLower(c.{name_prop}) IN $words
           OR toLower(c.{formula_prop}) IN $formulas
           OR any(alias IN c.aliases WHERE toLower(alias) IN $words)
        OPTIONAL MATCH (h:{herb_label})-[:{rel_type}]->(c)
        RETURN c.{name_prop} AS name,
               c.{formula_prop} AS formula,
               c.aliases AS aliases,
               collect(DISTINCT h.{herb_name_prop}) AS related_herbs,
               collect(DISTINCT h.{scientific_name_prop}) AS scientific_names
        LIMIT 10
        """

        def _execute_search():
            with neo4j_driver.session() as session:
                res = session.run(cypher, words=all_words, formulas=search_formulas)
                return [dict(r) for r in res]

        records = await loop.run_in_executor(None, _execute_search)

        # Fallback to loose fuzzy lookup if nothing matched
        if not records and all_words:
            loose_cypher = f"""
            MATCH (c:{compound_label})
            WHERE any(w IN $words WHERE toLower(c.{name_prop}) CONTAINS w)
            OPTIONAL MATCH (h:{herb_label})-[:{rel_type}]->(c)
            RETURN c.{name_prop} AS name,
                   c.{formula_prop} AS formula,
                   c.aliases AS aliases,
                   collect(DISTINCT h.{herb_name_prop}) AS related_herbs,
                   collect(DISTINCT h.{scientific_name_prop}) AS scientific_names
            LIMIT 5
            """
            def _execute_loose_search():
                with neo4j_driver.session() as session:
                    res = session.run(loose_cypher, words=all_words)
                    return [dict(r) for r in res]
            records = await loop.run_in_executor(None, _execute_loose_search)

        candidates = []
        for rec in records:
            score = 0.0
            matched_evidence = []

            c_name = rec.get("name") or ""
            c_formula = rec.get("formula") or ""
            c_aliases = [a.lower() for a in rec.get("aliases") or []]
            related_herbs = rec.get("related_herbs") or []
            scientific_names = rec.get("scientific_names") or []

            # 1. Exact Compound Name
            if any(term in [c_name.lower(), c_name.lower().replace("-", "")] for term in search_terms):
                score += 0.45
                matched_evidence.append("Exact compound name match")
            # 2. Alias match
            elif any(alias in search_terms for alias in c_aliases):
                score += 0.20
                matched_evidence.append("Alias match")
            # 3. Fuzzy search word match
            elif any(w in c_name.lower() for w in all_words):
                score += 0.10
                matched_evidence.append("Fuzzy name match")

            # 4. Exact molecular formula
            if c_formula and any(f == c_formula.lower() for f in search_formulas):
                score += 0.30
                matched_evidence.append("Exact molecular formula match")
            # Contradicting formula (if OCR formula found, but it differs from DB candidate formula)
            elif c_formula and search_formulas:
                score -= 0.50
                matched_evidence.append("Contradicting formula")

            # 5. Related Herb Match
            # Check if related herb common or scientific name mentioned in query or raw OCR
            flat_herb_names = [h.lower() for h in related_herbs + scientific_names]
            if any(any(h in text for h in flat_herb_names) for text in [extraction.raw_text.lower(), user_query.lower()]):
                score += 0.15
                matched_evidence.append("Related herb match")

            # 6. Visible label consistency (e.g. OH, CH3 in OCR struct list matches candidate properties or aliases)
            if extraction.visible_labels:
                # heuristic: structural groups presence
                groups_matched = False
                for group in extraction.visible_labels:
                    if group.lower() in c_name.lower() or any(group.lower() in a for a in c_aliases):
                        groups_matched = True
                if groups_matched:
                    score += 0.10
                    matched_evidence.append("Visible label consistency")

            # Clamp score between 0.0 and 1.0
            normalized_score = max(0.0, min(1.0, score))

            candidates.append(
                Neo4jCandidate(
                    entity_type="Compound",
                    name=c_name,
                    scientific_name=scientific_names[0] if scientific_names else None,
                    formula=c_formula or None,
                    related_herbs=related_herbs,
                    matched_evidence=matched_evidence,
                    score=normalized_score,
                )
            )

        # Sort candidates by score descending
        candidates.sort(key=lambda x: x.score, reverse=True)

        # Resolve overall status
        status = "insufficient_evidence"
        overall_confidence = 0.0
        limitations = []

        if candidates:
            best_candidate = candidates[0]
            overall_confidence = best_candidate.score

            if overall_confidence >= settings.ATTACHMENT_HIGH_CONFIDENCE:
                status = "verified"
            elif overall_confidence >= settings.ATTACHMENT_MIN_CONFIDENCE:
                status = "partially_verified"
            elif overall_confidence > 0.30:
                status = "ambiguous"
            else:
                status = "insufficient_evidence"
        else:
            limitations.append("Tidak ditemukan kandidat senyawa yang cocok di database grafik.")

        if extraction.detected_type == "chemical_structure_diagram" and status != "verified":
            limitations.append("Identifikasi skeletal structure tanpa data penamaan tervalidasi (SMILES/InChI) memiliki tingkat ketidakpastian tinggi.")

        return Neo4jVerificationResult(
            success=True,
            verification_status=status,
            candidates=candidates,
            confidence=overall_confidence,
            limitations=limitations,
        )

    except Exception as err:
        logger.error(f"Neo4j verification query failed: {err}", exc_info=True)
        return Neo4jVerificationResult(
            success=False,
            verification_status="failed",
            warnings=[f"neo4j_error: {str(err)}"],
            limitations=["Verifikasi database graph (Neo4j) sedang tidak tersedia saat ini."],
        )

