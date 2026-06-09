from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.core.database import neo4j_driver, supabase
from app.core.dependencies import ModelTier, Persona

PLANT_PART_WORDS = {
    "daun", "buah", "akar", "rimpang", "batang", "bunga", "biji", "kulit",
    "herba", "umbi", "getah", "minyak", "ekstrak", "simplisia",
}
STOPWORDS = {
    "apa", "aja", "saja", "dan", "di", "dalam", "yang", "untuk", "apa", "kegunaannya",
    "manfaat", "khasiat", "senyawa", "aktif", "kandungan", "jelaskan", "tolong",
}
PLANT_ALIASES: dict[str, dict[str, Any]] = {
    "kelor": {
        "canonical_local_name": "kelor",
        "scientific_name": "Moringa oleifera",
        "family": "Moringaceae",
        "synonyms": ["daun kelor", "kelor", "moringa", "Moringa oleifera"],
    },
    "daun kelor": {
        "canonical_local_name": "kelor",
        "scientific_name": "Moringa oleifera",
        "family": "Moringaceae",
        "synonyms": ["daun kelor", "kelor", "moringa", "Moringa oleifera"],
    },
    "moringa": {
        "canonical_local_name": "kelor",
        "scientific_name": "Moringa oleifera",
        "family": "Moringaceae",
        "synonyms": ["daun kelor", "kelor", "moringa", "Moringa oleifera"],
    },
}
FORBIDDEN_KELOR_MISMATCHES = {
    "pyrrosia petiolosa",
    "pyrrosia elliptica",
    "pyrrosia lingua",
    "graptophyllum pictum",
}
IDENTITY_KEYS = {
    "entity_id", "plant_id", "canonical_id", "id", "element_id",
    "scientific_name", "nama_latin", "latinName", "latin_name",
    "local_name", "nama", "tanaman", "commonName", "common_name", "topik",
}


class CanonicalPlantIdentity(BaseModel):
    original_query: str
    extracted_local_name: str | None = None
    canonical_local_name: str | None = None
    scientific_name: str | None = None
    family: str | None = None
    synonyms: list[str] = Field(default_factory=list)
    entity_id: str | None = None
    confidence: float = 0.0
    resolution_method: Literal[
        "exact_database_match", "exact_alias_match", "scientific_name_match",
        "normalized_local_name_match", "high_confidence_fuzzy_match",
        "ambiguous", "not_found",
    ]
    evidence_sources: list[str] = Field(default_factory=list)
    ambiguous_candidates: list[dict[str, Any]] = Field(default_factory=list)


class EntityLock(BaseModel):
    entity_id: str | None = None
    scientific_name: str | None = None
    canonical_local_name: str | None = None
    allowed_synonyms: list[str] = Field(default_factory=list)


class EvidenceItem(BaseModel):
    claim_type: str = "general"
    subject: str = ""
    predicate: str = ""
    value: str = ""
    source: str = "retrieval"
    source_type: str = "unknown"
    evidence_level: str = "insufficient_evidence"
    entity_id: str | None = None
    scientific_name: str | None = None
    confidence: float = 0.0
    verified: bool = False


class ContextConflict(BaseModel):
    field: str
    values: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    severity: str = "low"


class GroundedContext(BaseModel):
    identity: CanonicalPlantIdentity
    plant_specific_evidence: list[dict[str, Any]] = Field(default_factory=list)
    general_evidence: list[dict[str, Any]] = Field(default_factory=list)
    rejected_records: list[dict[str, Any]] = Field(default_factory=list)
    conflicts: list[ContextConflict] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    retrieval_metadata: dict[str, Any] = Field(default_factory=dict)
    evidence_items: list[EvidenceItem] = Field(default_factory=list)

    def to_prompt_text(self) -> str:
        lines = [
            "[CANONICAL PLANT IDENTITY]",
            f"local_name: {self.identity.canonical_local_name or self.identity.extracted_local_name or 'not_found'}",
            f"scientific_name: {self.identity.scientific_name or 'not_found'}",
            f"family: {self.identity.family or 'not_available'}",
            f"confidence: {self.identity.confidence:.2f}",
            f"resolution_method: {self.identity.resolution_method}",
            "",
            "[VERIFIED PLANT-SPECIFIC EVIDENCE]",
        ]
        lines.extend(_format_record(i, rec) for i, rec in enumerate(self.plant_specific_evidence, 1))
        if not self.plant_specific_evidence:
            lines.append("Data spesifik tanaman belum tersedia dari retrieval terverifikasi.")

        # Include structured EvidenceItems mapping
        lines.append("\n[EVIDENCE ITEMS]")
        for i, item in enumerate(self.evidence_items, 1):
            lines.append(f"- Item #{i}: claim_type={item.claim_type} subject={item.subject} predicate={item.predicate} value={item.value} evidence_level={item.evidence_level} scientific_name={item.scientific_name} verified={item.verified}")
        if not self.evidence_items:
            lines.append("Tidak ada evidence item terpetakan.")

        lines.append("\n[GENERAL DOMAIN EVIDENCE]")
        lines.extend(_format_record(i, rec) for i, rec in enumerate(self.general_evidence, 1))
        if not self.general_evidence:
            lines.append("Tidak ada evidence umum tambahan.")
        lines.append("\n[CONFLICTING RECORDS REMOVED]")
        for rec in self.rejected_records[:12]:
            lines.append(f"- removed_source={rec.get('source_type', 'unknown')} reason={rec.get('rejection_reason', 'entity_mismatch')} scientific_name={_record_scientific_name(rec) or 'unknown'}")
        if not self.rejected_records:
            lines.append("Tidak ada record konflik yang dibuang.")
        lines.append("\n[DATA LIMITATIONS]")
        lines.extend(f"- {item}" for item in (self.limitations or ["Tidak ada keterbatasan tambahan terdeteksi."]))
        return "\n".join(lines)


class ValidationResult(BaseModel):
    passed: bool
    checks: dict[str, bool] = Field(default_factory=dict)
    reasons: list[str] = Field(default_factory=list)


def normalize_text(value: Any, *, strip_parts: bool = False) -> str:
    text = re.sub(r"[^\w\s.-]", " ", str(value or "").lower())
    tokens = [t for t in text.split() if t]
    if strip_parts:
        tokens = [t for t in tokens if t not in PLANT_PART_WORDS and t not in STOPWORDS]
    return " ".join(tokens).strip()


def extract_plant_phrase(query: str) -> str | None:
    normalized = normalize_text(query)
    for alias in sorted(PLANT_ALIASES, key=len, reverse=True):
        if alias in normalized:
            return alias
    tokens = [t for t in normalize_text(query, strip_parts=True).split() if len(t) > 2]
    return " ".join(tokens[:2]) if tokens else None


def resolve_canonical_plant_identity(query: str) -> CanonicalPlantIdentity:
    extracted = extract_plant_phrase(query)

    # 1. Deterministic Alias Registry Check First
    alias_key = normalize_text(extracted, strip_parts=True) if extracted else None
    if alias_key in PLANT_ALIASES:
        data = PLANT_ALIASES[alias_key]
        return CanonicalPlantIdentity(
            original_query=query,
            extracted_local_name=extracted,
            canonical_local_name=data["canonical_local_name"],
            scientific_name=data["scientific_name"],
            family=data.get("family"),
            synonyms=data.get("synonyms", []),
            confidence=0.98,
            resolution_method="exact_alias_match",
            evidence_sources=["PLANT_ALIASES"],
        )

    # 2. Database Exact Matches (Neo4j first, then Supabase fallback)
    db_identity = _resolve_from_neo4j(query, extracted)
    if db_identity and db_identity.resolution_method != "ambiguous":
        return db_identity

    sb_identity = _resolve_from_supabase(query, extracted)
    if sb_identity and sb_identity.resolution_method != "ambiguous":
        return sb_identity

    if db_identity and db_identity.resolution_method == "ambiguous":
        return db_identity
    if sb_identity and sb_identity.resolution_method == "ambiguous":
        return sb_identity

    # 3. High-Confidence Fuzzy Matching (Neo4j first, then PLANT_ALIASES)
    fuzzy_db = _resolve_fuzzy_from_db(query, extracted)
    if fuzzy_db:
        return fuzzy_db

    fuzzy = _resolve_alias_fuzzy(query, extracted)
    if fuzzy:
        return fuzzy

    return CanonicalPlantIdentity(
        original_query=query,
        extracted_local_name=extracted,
        resolution_method="not_found",
        confidence=0.0,
        evidence_sources=[],
        ambiguous_candidates=[],
    )


def _get_binomial_name(scientific_name: str | None) -> str | None:
    if not scientific_name:
        return None
    words = re.findall(r'\b[A-Z][a-z]+\s+[a-z]+\b', scientific_name.strip())
    if words:
        return words[0]
    parts = scientific_name.strip().split()
    if len(parts) >= 2:
        return f"{parts[0]} {parts[1]}"
    return scientific_name


def _resolve_from_neo4j(query: str, extracted: str | None) -> CanonicalPlantIdentity | None:
    search_terms = [term for term in {extracted, normalize_text(extracted, strip_parts=True), normalize_text(query, strip_parts=True)} if term]
    search_terms = [t for t in search_terms if t not in STOPWORDS]
    if not search_terms:
        return None
    cypher = """
    MATCH (h:Herb)
    WHERE any(term IN $terms WHERE
        toLower(h.commonName) = term OR
        toLower(h.latinName) = term OR
        term IN [x IN coalesce(h.localNames, []) | toLower(toString(x))]
    )
    OPTIONAL MATCH (h)-[:BELONGS_TO]->(f:Family)
    RETURN elementId(h) AS entity_id, h.commonName AS commonName, h.latinName AS latinName,
           f.name AS family, h.localNames AS localNames
    LIMIT 3
    """
    try:
        records, _, _ = neo4j_driver.execute_query(cypher, terms=search_terms)
    except Exception:
        return None
    if not records:
        return None
    if len(records) > 1:
        scis = {normalize_text(_get_binomial_name(r.data().get("latinName") or r.data().get("scientific_name"))) for r in records}
        scis = {s for s in scis if s}
        if len(scis) == 1:
            rec = records[0].data()
            return CanonicalPlantIdentity(
                original_query=query,
                extracted_local_name=extracted,
                canonical_local_name=(rec.get("commonName") or extracted).lower(),
                scientific_name=_get_binomial_name(rec.get("latinName")),
                family=rec.get("family"),
                synonyms=list(rec.get("localNames") or []),
                entity_id=rec.get("entity_id"),
                confidence=0.95,
                resolution_method="exact_database_match",
                evidence_sources=["Neo4j exact query (deduplicated)"],
            )

        candidates = [r.data() for r in records]
        return CanonicalPlantIdentity(
            original_query=query,
            extracted_local_name=extracted,
            resolution_method="ambiguous",
            confidence=0.4,
            evidence_sources=["Neo4j exact query"],
            ambiguous_candidates=candidates,
        )
    rec = records[0].data()
    return CanonicalPlantIdentity(
        original_query=query,
        extracted_local_name=extracted,
        canonical_local_name=(rec.get("commonName") or extracted).lower(),
        scientific_name=_get_binomial_name(rec.get("latinName")),
        family=rec.get("family"),
        synonyms=list(rec.get("localNames") or []),
        entity_id=rec.get("entity_id"),
        confidence=0.95,
        resolution_method="exact_database_match",
        evidence_sources=["Neo4j exact query"],
    )


def _resolve_from_supabase(query: str, extracted: str | None) -> CanonicalPlantIdentity | None:
    search_terms = [term for term in {extracted, normalize_text(extracted, strip_parts=True), normalize_text(query, strip_parts=True)} if term]
    search_terms = [t for t in search_terms if t not in STOPWORDS]
    if not search_terms:
        return None
    try:
        for term in search_terms:
            res = supabase.table("plants").select("id, nama, nama_latin, famili").or_(f"nama.ilike.{term},nama_latin.ilike.{term}").execute()
            if res.data:
                if len(res.data) > 1:
                    scis = {normalize_text(_get_binomial_name(r.get("nama_latin") or r.get("scientific_name"))) for r in res.data}
                    scis = {s for s in scis if s}
                    if len(scis) == 1:
                        rec = res.data[0]
                    else:
                        return CanonicalPlantIdentity(
                            original_query=query,
                            extracted_local_name=extracted,
                            resolution_method="ambiguous",
                            confidence=0.4,
                            evidence_sources=["Supabase exact query"],
                            ambiguous_candidates=res.data,
                        )
                else:
                    rec = res.data[0]
                return CanonicalPlantIdentity(
                    original_query=query,
                    extracted_local_name=extracted,
                    canonical_local_name=(rec.get("nama") or extracted).lower(),
                    scientific_name=_get_binomial_name(rec.get("nama_latin")),
                    family=rec.get("famili"),
                    synonyms=[rec.get("nama")] if rec.get("nama") else [],
                    entity_id=str(rec.get("id")),
                    confidence=0.95,
                    resolution_method="exact_database_match",
                    evidence_sources=["Supabase exact query"],
                )
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"Failed to resolve from Supabase: {e}")
    return None


def _resolve_fuzzy_from_db(query: str, extracted: str | None) -> CanonicalPlantIdentity | None:
    target = normalize_text(extracted or query, strip_parts=True)
    if not target:
        return None
    cypher = "MATCH (h:Herb) RETURN elementId(h) AS id, h.commonName AS commonName, h.latinName AS latinName, h.localNames AS localNames"
    try:
        records, _, _ = neo4j_driver.execute_query(cypher)
    except Exception:
        records = []

    candidates = []
    for r in records:
        data = r.data()
        names = {data.get("commonName"), data.get("latinName"), *(data.get("localNames") or [])}
        names = {normalize_text(n, strip_parts=True) for n in names if n}

        best_score = 0.0
        for name in names:
            score = SequenceMatcher(None, target, name).ratio()
            if score > best_score:
                best_score = score

        if best_score >= 0.90:
            candidates.append((best_score, data))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0], reverse=True)
    if len(candidates) > 1 and candidates[0][0] - candidates[1][0] < 0.05:
        diff_scis = {c[1].get("latinName") for c in candidates[:3]}
        if len(diff_scis) > 1:
            return CanonicalPlantIdentity(
                original_query=query,
                extracted_local_name=extracted,
                resolution_method="ambiguous",
                confidence=candidates[0][0],
                ambiguous_candidates=[c[1] for c in candidates[:3]],
            )

    best_score, rec = candidates[0]
    return CanonicalPlantIdentity(
        original_query=query,
        extracted_local_name=extracted,
        canonical_local_name=(rec.get("commonName") or extracted).lower(),
        scientific_name=_get_binomial_name(rec.get("latinName")),
        synonyms=list(rec.get("localNames") or []),
        entity_id=rec.get("id"),
        confidence=best_score,
        resolution_method="high_confidence_fuzzy_match",
        evidence_sources=["Neo4j fuzzy query"],
    )


def _resolve_alias_fuzzy(query: str, extracted: str | None) -> CanonicalPlantIdentity | None:
    target = normalize_text(extracted or query, strip_parts=True)
    if not target:
        return None
    scored = sorted(
        ((SequenceMatcher(None, target, alias).ratio(), alias) for alias in PLANT_ALIASES),
        reverse=True,
    )
    if not scored or scored[0][0] < 0.92:
        return None
    if len(scored) > 1 and scored[0][0] - scored[1][0] < 0.03:
        return CanonicalPlantIdentity(
            original_query=query,
            extracted_local_name=extracted,
            resolution_method="ambiguous",
            confidence=scored[0][0],
            ambiguous_candidates=[{"alias": alias, "score": score} for score, alias in scored[:3]],
        )
    data = PLANT_ALIASES[scored[0][1]]
    return CanonicalPlantIdentity(
        original_query=query,
        extracted_local_name=extracted,
        canonical_local_name=data["canonical_local_name"],
        scientific_name=data["scientific_name"],
        family=data.get("family"),
        synonyms=data.get("synonyms", []),
        confidence=scored[0][0],
        resolution_method="high_confidence_fuzzy_match",
        evidence_sources=["PLANT_ALIASES fuzzy"],
    )


def build_entity_lock(identity: CanonicalPlantIdentity) -> EntityLock:
    synonyms = [identity.canonical_local_name, identity.extracted_local_name, identity.scientific_name, *identity.synonyms]
    return EntityLock(
        entity_id=identity.entity_id,
        scientific_name=identity.scientific_name,
        canonical_local_name=identity.canonical_local_name,
        allowed_synonyms=list(dict.fromkeys(s for s in synonyms if s)),
    )


def _is_different_plant(record: dict[str, Any], identity: CanonicalPlantIdentity) -> bool:
    if identity.resolution_method in {"not_found", "ambiguous"}:
        return False

    # Check scientific name mismatch
    rec_sci = _record_scientific_name(record)
    if rec_sci and identity.scientific_name:
        if normalize_text(rec_sci) != normalize_text(identity.scientific_name):
            return True

    # Check local/common name mismatch
    rec_local = _record_local_name(record)
    if rec_local:
        lock_names = {normalize_text(n, strip_parts=True) for n in [identity.canonical_local_name, identity.extracted_local_name, *identity.synonyms] if n}
        norm_rec_local = normalize_text(rec_local, strip_parts=True)
        if norm_rec_local and norm_rec_local not in lock_names:
            for alias, data in PLANT_ALIASES.items():
                if normalize_text(alias, strip_parts=True) == norm_rec_local:
                    if data["scientific_name"] != identity.scientific_name:
                        return True
            if identity.scientific_name and normalize_text(identity.scientific_name) == "moringa oleifera":
                for mismatch in FORBIDDEN_KELOR_MISMATCHES:
                    if mismatch in norm_rec_local or norm_rec_local in mismatch:
                        return True
    return False


def _map_record_to_evidence_items(record: dict[str, Any], identity: CanonicalPlantIdentity) -> list[EvidenceItem]:
    items = []
    sci_name = identity.scientific_name or _record_scientific_name(record)
    ent_id = identity.entity_id or record.get("entity_id") or record.get("id")
    local_name = identity.canonical_local_name or identity.extracted_local_name or _record_local_name(record) or "tanaman"

    default_level = record.get("evidence_level") or "insufficient_evidence"

    compounds = record.get("compounds") or []
    if isinstance(compounds, str):
        compounds = [c.strip() for c in compounds.split(",") if c.strip()]
    elif not isinstance(compounds, list):
        compounds = []

    kandungan = record.get("kandungan_kimia")
    if kandungan:
        compounds.append(kandungan)

    for comp in compounds:
        items.append(EvidenceItem(
            claim_type="phytochemical",
            subject=local_name,
            predicate="contains",
            value=str(comp),
            source="database",
            source_type=record.get("source_type", "unknown"),
            evidence_level="phytochemical_screening",
            entity_id=ent_id,
            scientific_name=sci_name,
            confidence=0.9,
            verified=True
        ))

    uses = record.get("traditional_uses") or []
    if isinstance(uses, str):
        uses = [u.strip() for u in uses.split(",") if u.strip()]
    elif not isinstance(uses, list):
        uses = []

    khasiat = record.get("khasiat")
    if khasiat:
        uses.append(khasiat)

    for use in uses:
        items.append(EvidenceItem(
            claim_type="therapeutic",
            subject=local_name,
            predicate="used_for",
            value=str(use),
            source="database",
            source_type=record.get("source_type", "unknown"),
            evidence_level=default_level if default_level != "insufficient_evidence" else "traditional",
            entity_id=ent_id,
            scientific_name=sci_name,
            confidence=0.8,
            verified=True
        ))

    if not items:
        desc = record.get("deskripsi") or record.get("konten") or ""
        if desc:
            items.append(EvidenceItem(
                claim_type="general",
                subject=local_name,
                predicate="description",
                value=str(desc)[:200],
                source="database",
                source_type=record.get("source_type", "unknown"),
                evidence_level=default_level,
                entity_id=ent_id,
                scientific_name=sci_name,
                confidence=0.7,
                verified=True
            ))

    return items


def filter_records_by_entity_lock(records: list[dict[str, Any]], identity: CanonicalPlantIdentity) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    accepted: list[dict[str, Any]] = []
    general: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for record in records:
        if record_matches_entity_lock(record, identity):
            accepted.append({**record, "source_scope": "plant_specific"})
        elif _is_different_plant(record, identity):
            rejected.append({**record, "rejection_reason": "entity_mismatch"})
        else:
            general.append({**record, "source_scope": "general_domain_knowledge"})
    return accepted, general, rejected


def record_matches_entity_lock(record: dict[str, Any], identity: CanonicalPlantIdentity) -> bool:
    if identity.resolution_method in {"not_found", "ambiguous"}:
        return False
    lock = build_entity_lock(identity)
    record_ids = {normalize_text(record.get(k)) for k in ("entity_id", "plant_id", "canonical_id", "id", "element_id") if record.get(k)}
    if lock.entity_id and normalize_text(lock.entity_id) in record_ids:
        return True
    sci = normalize_text(_record_scientific_name(record))
    if lock.scientific_name and sci and sci == normalize_text(lock.scientific_name):
        return True
    names = {_record_local_name(record), *(_as_list(record.get("synonyms"))), *(_as_list(record.get("localNames")))}
    norm_names = {normalize_text(name, strip_parts=True) for name in names if name}
    allowed = {normalize_text(name, strip_parts=True) for name in lock.allowed_synonyms if name}
    return bool(norm_names & allowed)


def build_grounded_context(
    *,
    query: str,
    identity: CanonicalPlantIdentity,
    vector_records: list[dict[str, Any]],
    graph_records: list[dict[str, Any]],
    persona: Persona,
    tier: ModelTier,
) -> GroundedContext:
    raw_records = [*_tag_records(vector_records, "vector"), *_tag_records(graph_records, "graph")]
    accepted, general, rejected = filter_records_by_entity_lock(raw_records, identity)
    max_plant = 6 if tier == ModelTier.FAST else 12
    max_general = 2 if tier == ModelTier.FAST else 4
    conflicts = _detect_conflicts(raw_records, identity, rejected)

    # Map accepted records to EvidenceItems
    evidence_items = []
    for rec in accepted[:max_plant]:
        evidence_items.extend(_map_record_to_evidence_items(rec, identity))

    limitations: list[str] = []
    if identity.resolution_method == "not_found":
        limitations.append("Identitas tanaman spesifik tidak ditemukan; sistem tidak akan mengganti dengan tanaman lain.")
    if identity.resolution_method == "ambiguous":
        limitations.append("Nama tanaman ambigu; klarifikasi diperlukan sebelum membuat klaim spesifik.")
    if not accepted:
        limitations.append("Tidak ada record plant-specific yang lolos entity lock.")
    return GroundedContext(
        identity=identity,
        plant_specific_evidence=accepted[:max_plant],
        general_evidence=general[:max_general],
        rejected_records=rejected,
        conflicts=conflicts,
        limitations=limitations,
        evidence_items=evidence_items,
        retrieval_metadata={
            "vector_count_raw": len(vector_records),
            "graph_count_raw": len(graph_records),
            "accepted_count": len(accepted),
            "general_count": len(general),
            "rejected_entity_mismatch": len(rejected),
        },
    )


def _detect_conflicts(records: list[dict[str, Any]], identity: CanonicalPlantIdentity, rejected: list[dict[str, Any]]) -> list[ContextConflict]:
    names = sorted({n for n in (_record_scientific_name(r) for r in records) if n})
    conflicts: list[ContextConflict] = []
    if len({normalize_text(n) for n in names}) > 1:
        conflicts.append(ContextConflict(field="scientific_name", values=names, sources=["retrieval"], severity="critical" if rejected else "medium"))
    return conflicts


def _format_record(index: int, rec: dict[str, Any]) -> str:
    safe = {k: v for k, v in rec.items() if k not in {"embedding"} and v not in (None, "", [])}
    return f"Evidence #{index}: " + "; ".join(f"{k}={v}" for k, v in list(safe.items())[:14])


def _tag_records(records: list[dict[str, Any]], source: str) -> list[dict[str, Any]]:
    return [{**rec, "source_type": rec.get("source_type") or source} for rec in records]


def _has_identity(record: dict[str, Any]) -> bool:
    return any(record.get(k) for k in IDENTITY_KEYS)


def _record_scientific_name(record: dict[str, Any]) -> str | None:
    for key in ("scientific_name", "nama_latin", "latinName", "latin_name"):
        if record.get(key):
            return str(record[key])
    return None


def _record_local_name(record: dict[str, Any]) -> str | None:
    for key in ("local_name", "nama", "tanaman", "commonName", "common_name", "topik"):
        if record.get(key):
            return str(record[key])
    return None


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value if v]
    return [str(value)]
