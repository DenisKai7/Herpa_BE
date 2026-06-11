"""Availability, safety scoring, and deterministic ranking for herbal candidates."""

from typing import Any

from app.agent.herbal_candidate_utils import build_canonical_herb_key_from_values, deduplicate_herbal_candidates
from app.agent.herbal_safety import medical_attention_signs, safety_assess
from app.models.herbal_recommendation import (
    AvailabilityInfo,
    EvidenceInfo,
    GraphProvenance,
    HerbalCandidate,
    HerbalRecommendationRequest,
    IngredientItem,
    PreparationMethod,
    SafetyItem,
    SourceProvenanceItem,
    UsageRule,
    VerifiedSafetyField,
)

AVAILABILITY_LABELS = {
    "easy_to_find": "Mudah dicari",
    "moderately_available": "Cukup mudah dicari",
    "hard_to_find": "Lebih sulit ditemukan",
    "seasonal": "Musiman",
    "restricted": "Terbatas",
    "unknown": "Ketersediaan belum diketahui",
}

EVIDENCE_LEVELS = {
    "traditional",
    "phytochemical_screening",
    "in_vitro",
    "in_vivo",
    "clinical",
    "systematic_review",
    "data_not_available",
    "insufficient_evidence",
}


def _clean_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    text = str(value).strip()
    return [text] if text else []


def _source_ids(item: dict[str, Any], fallback: list[str]) -> list[str]:
    values = item.get("source_ids") or item.get("sourceIds") or item.get("source_id") or item.get("source")
    if values:
        return _clean_list(values)
    return fallback.copy()


def _first_source(source_ids: list[str]) -> str | None:
    return source_ids[0] if source_ids else None


def classify_availability(raw: dict[str, Any]) -> tuple[str, str, str | None, float]:
    value = raw.get("availability")
    score = raw.get("availability_score")
    reason = raw.get("availability_reason")
    normalized = str(value).lower().strip() if value else ""
    mapping = {
        "easy": "easy_to_find",
        "easy_to_find": "easy_to_find",
        "mudah": "easy_to_find",
        "moderate": "moderately_available",
        "moderately_available": "moderately_available",
        "cukup": "moderately_available",
        "hard": "hard_to_find",
        "hard_to_find": "hard_to_find",
        "sulit": "hard_to_find",
        "seasonal": "seasonal",
        "restricted": "restricted",
        "unknown": "unknown",
    }
    if normalized in mapping:
        availability = mapping[normalized]
    else:
        try:
            numeric = float(score)
        except (TypeError, ValueError):
            numeric = None
        if numeric is None:
            availability = "unknown"
        elif numeric >= 0.75:
            availability = "easy_to_find"
        elif numeric >= 0.40:
            availability = "moderately_available"
        else:
            availability = "hard_to_find"
    availability_score = {
        "easy_to_find": 1.0,
        "moderately_available": 0.65,
        "hard_to_find": 0.25,
        "seasonal": 0.5,
        "restricted": 0.2,
        "unknown": 0.0,
    }[availability]
    return availability, AVAILABILITY_LABELS[availability], reason, availability_score


def _ingredient_items(value: Any, fallback_sources: list[str]) -> list[IngredientItem]:
    if not isinstance(value, list):
        return []
    result: list[IngredientItem] = []
    for idx, item in enumerate(value):
        if isinstance(item, dict):
            name = str(item.get("name") or item.get("ingredient") or "").strip()
            sources = _source_ids(item, fallback_sources)
            if name:
                result.append(IngredientItem(name=name, amount_text=item.get("amount_text") or item.get("amountText"), source_ids=sources))
        elif str(item).strip():
            result.append(IngredientItem(name=str(item).strip(), amount_text=None, source_ids=fallback_sources.copy()))
    return result


def _prep_methods(raw: dict[str, Any], herb_id: str, fallback_sources: list[str]) -> list[PreparationMethod]:
    methods = raw.get("preparation_methods") or []
    if not isinstance(methods, list):
        return []
    result = []
    for idx, item in enumerate(methods):
        if not isinstance(item, dict):
            continue
        sources = _source_ids(item, fallback_sources)
        title = str(item.get("title") or item.get("name") or "").strip()
        steps = _clean_list(item.get("steps"))
        ingredients = _ingredient_items(item.get("ingredients"), sources)
        if not title and not steps and not ingredients:
            continue
        plant_part = item.get("plant_part") or item.get("plantPart")
        dosage_form = item.get("dosage_form") or item.get("dosageForm") or item.get("preparation_type") or item.get("type") or item.get("form")
        suitable = _clean_list(item.get("suitable_symptoms") or item.get("suitableSymptoms") or item.get("compatible_symptoms"))
        result.append(PreparationMethod(
            method_id=str(item.get("method_id") or item.get("methodId") or item.get("id") or f"{herb_id}-prep-{idx}"),
            title=title,
            plant_part=str(plant_part or ""),
            dosage_form=str(dosage_form or ""),
            ingredients=ingredients,
            steps=steps,
            water_volume_text=item.get("water_volume_text") or item.get("waterVolumeText"),
            temperature_text=item.get("temperature_text") or item.get("temperatureText"),
            preparation_duration_text=item.get("preparation_duration_text") or item.get("preparationDurationText"),
            storage_instruction=item.get("storage_instruction") or item.get("storageInstruction"),
            suitable_symptoms=suitable,
            evidence_level=str(item.get("evidence_level") or item.get("evidenceLevel") or "data_not_available"),
            verification_status=str(item.get("verification_status") or item.get("verificationStatus") or "unverified"),
            source_ids=sources,
            preparation_type=str(item.get("preparation_type") or item.get("type") or dosage_form or "data_not_available"),
            source=_first_source(sources),
            compatible_symptoms=suitable,
            contraindicated_groups=_clean_list(item.get("contraindicated_groups") or item.get("contraindicatedGroups")),
        ))
    return result


def _usage_rules(raw: dict[str, Any], fallback_sources: list[str]) -> list[UsageRule]:
    rules = raw.get("usage_rules") or []
    if not isinstance(rules, list):
        return []
    result = []
    for idx, item in enumerate(rules):
        if not isinstance(item, dict):
            continue
        if not any(item.get(k) for k in ["form", "amount_text", "amountText", "frequency_text", "frequencyText", "duration_text", "durationText"]):
            continue
        sources = _source_ids(item, fallback_sources)
        allowed = _clean_list(item.get("allowed_age_groups") or item.get("allowedAgeGroups") or item.get("applicable_age_groups"))
        result.append(UsageRule(
            usage_rule_id=str(item.get("usage_rule_id") or item.get("usageRuleId") or item.get("id") or f"usage-{idx}"),
            form=item.get("form"),
            amount_text=str(item.get("amount_text") or item.get("amountText") or ""),
            frequency_text=str(item.get("frequency_text") or item.get("frequencyText") or ""),
            administration_time_text=item.get("administration_time_text") or item.get("administrationTimeText"),
            duration_text=str(item.get("duration_text") or item.get("durationText") or ""),
            maximum_duration_text=item.get("maximum_duration_text") or item.get("maximumDurationText"),
            before_or_after_meal=item.get("before_or_after_meal") or item.get("beforeOrAfterMeal"),
            administration_notes=_clean_list(item.get("administration_notes") or item.get("administrationNotes")),
            allowed_age_groups=allowed,
            prohibited_age_groups=_clean_list(item.get("prohibited_age_groups") or item.get("prohibitedAgeGroups")),
            applicable_age_groups=allowed,
            evidence_level=str(item.get("evidence_level") or item.get("evidenceLevel") or "data_not_available"),
            verification_status=str(item.get("verification_status") or item.get("verificationStatus") or "unverified"),
            source_ids=sources,
            source=_first_source(sources),
        ))
    return result


def _safety_item(item: Any, fallback_sources: list[str], prefix: str, idx: int) -> SafetyItem | None:
    if isinstance(item, dict):
        sources = _source_ids(item, fallback_sources)
        title = str(item.get("title") or item.get("name") or item.get("label") or "").strip()
        description = str(item.get("description") or title).strip()
        action = str(item.get("action_text") or item.get("actionText") or item.get("action") or "Ikuti arahan pada sumber terverifikasi.").strip()
        severity = str(item.get("severity") or "unspecified").strip()
        safety_id = str(item.get("safety_id") or item.get("id") or f"{prefix}-{idx}")
        if title and sources:
            return SafetyItem(safety_id=safety_id, id=safety_id, label=title, title=title, description=description, severity=severity, action_text=action, source_ids=sources)
        return None
    text = str(item or "").strip()
    if not text or not fallback_sources:
        return None
    return SafetyItem(safety_id=f"{prefix}-{idx}", id=f"{prefix}-{idx}", label=text, title=text, description=text, severity="unspecified", action_text="Ikuti arahan pada sumber terverifikasi.", source_ids=fallback_sources.copy())


def _safety_field(raw: dict[str, Any], key: str, fallback_sources: list[str]) -> VerifiedSafetyField:
    section = raw.get(f"{key}_status") or raw.get(f"{key}Status")
    values = raw.get(key) or []
    if isinstance(section, dict):
        status = str(section.get("status") or "missing")
        source_ids = _clean_list(section.get("source_ids") or section.get("sourceIds") or section.get("source_id") or section.get("source"))
        item_values = section.get("items") if isinstance(section.get("items"), list) else values
        items = [item for idx, value in enumerate(item_values or []) if (item := _safety_item(value, source_ids, key, idx))]
        return VerifiedSafetyField(status=status, items=items, source_ids=source_ids, verified_at=section.get("verified_at") or section.get("verifiedAt"))
    if not isinstance(values, list):
        values = [values]
    items = [item for idx, value in enumerate(values) if (item := _safety_item(value, fallback_sources, key, idx))]
    if items:
        return VerifiedSafetyField(status="known_issue", items=items, source_ids=list(dict.fromkeys([s for item in items for s in item.source_ids])))
    if raw.get(f"{key}_no_known_issue") is True and fallback_sources:
        return VerifiedSafetyField(status="no_known_issue_within_source_scope", source_ids=fallback_sources.copy())
    return VerifiedSafetyField(status="missing")


def _sources(raw: dict[str, Any]) -> list[SourceProvenanceItem]:
    result = []
    for item in raw.get("sources", []) or []:
        if not isinstance(item, dict):
            continue
        source_id = str(item.get("source_id") or item.get("id") or "").strip()
        title = str(item.get("title") or item.get("name") or "").strip()
        if not source_id or not title:
            continue
        result.append(SourceProvenanceItem(
            source_id=source_id,
            title=title,
            publisher=item.get("publisher"),
            year=item.get("year"),
            evidence_grade=item.get("evidence_grade") or item.get("qualityGrade") or item.get("quality_grade"),
            url=item.get("url"),
            verified_at=item.get("verified_at") or item.get("verifiedAt") or item.get("accessDate"),
            active=bool(item.get("active", True)),
        ))
    return result


def _reason(local_name: str, matched: list[str], uses: list[str], evidence_level: str) -> str:
    symptom_text = ", ".join(matched or uses)
    if symptom_text:
        return f"{local_name} direkomendasikan karena memiliki penggunaan terverifikasi pada knowledge graph untuk {symptom_text} dengan tingkat bukti {evidence_level}."
    return f"{local_name} direkomendasikan berdasarkan penggunaan terverifikasi pada knowledge graph dengan tingkat bukti {evidence_level}."


def score_candidate(raw: dict[str, Any], symptoms: list[str], req: HerbalRecommendationRequest) -> HerbalCandidate:
    herb_id = str(raw.get("herb_id") or raw.get("database_id") or raw.get("local_name") or "unknown")
    local_name = str(raw.get("local_name") or "Tanaman belum teridentifikasi")
    scientific_name = raw.get("scientific_name")
    canonical_key = build_canonical_herb_key_from_values(herb_id, scientific_name, local_name)
    matched = list(dict.fromkeys(raw.get("matched_symptoms", [])))
    unmatched = [s for s in symptoms if s.lower() not in " | ".join(matched).lower()]
    availability, label, reason, availability_component = classify_availability(raw)
    safety_status, safety_reasons = safety_assess(raw, req)
    source_ids = [str(v) for v in raw.get("source_ids", []) if v]
    prep = _prep_methods(raw, herb_id, source_ids)
    usage = _usage_rules(raw, source_ids)
    evidence_level = str(raw.get("evidence_level") or "data_not_available")
    if evidence_level not in EVIDENCE_LEVELS:
        evidence_level = "insufficient_evidence"

    warnings = [item for idx, value in enumerate(raw.get("warnings") or []) if (item := _safety_item(value, source_ids, "warning", idx))]
    graph_verified = bool(source_ids and raw.get("graph_node_ids"))
    source_details = _sources(raw)
    provenance = GraphProvenance(
        graph_verified=graph_verified,
        coverage_score=1.0 if graph_verified else 0.0,
        source_ids=source_ids,
        sources=source_details,
        evidence_claim_ids=[str(v) for v in raw.get("evidence_claim_ids", []) if v],
        graph_node_ids=[str(v) for v in raw.get("graph_node_ids", []) if v],
        graph_relationship_ids=[str(v) for v in raw.get("graph_relationship_ids", []) if v],
        verified_at=raw.get("verified_at"),
        data_version=str(raw.get("data_version") or "herbal-recommendation-v1"),
    )
    availability_info = AvailabilityInfo(category=availability, label=label, reason=reason or "Data ketersediaan terverifikasi knowledge graph.", source_ids=source_ids) if graph_verified and availability != "unknown" else None
    evidence_info = EvidenceInfo(level=evidence_level, label=evidence_level, source_ids=source_ids) if graph_verified else None

    symptom_score = min(1.0, len(matched) / max(1, len(symptoms)))
    evidence_score = 0.8 if evidence_level in {"clinical", "systematic_review"} else 0.55 if evidence_level != "data_not_available" else 0.25
    safety_score = {"eligible": 1.0, "conditional": 0.55, "excluded": 0.0}[safety_status]
    prep_score = 1.0 if prep else 0.0
    usage_score = 1.0 if usage else 0.0
    safety_data_score = 1.0 if all(_safety_field(raw, key, source_ids).status != "missing" for key in ["contraindications", "interactions", "side_effects", "risk_groups"]) else 0.0
    traditional_score = 1.0 if raw.get("traditional_uses") else 0.0
    score = (
        symptom_score * 0.30 + evidence_score * 0.20 + safety_score * 0.15 +
        prep_score * 0.12 + usage_score * 0.10 + safety_data_score * 0.08 +
        availability_component * 0.03 + traditional_score * 0.02
    )
    if safety_status == "excluded":
        score -= 1.0

    plant_parts = list(dict.fromkeys([method.plant_part for method in prep if method.plant_part]))
    traditional_uses = raw.get("traditional_uses", [])
    return HerbalCandidate(
        herb_id=herb_id,
        canonical_key=canonical_key,
        source_herb_ids=[herb_id],
        local_name=local_name,
        scientific_name=scientific_name,
        aliases=raw.get("aliases", []),
        matched_symptoms=matched,
        unmatched_symptoms=unmatched,
        recommendation_reason=_reason(local_name, matched, traditional_uses, evidence_level),
        plant_parts=plant_parts,
        active_compounds=_clean_list(raw.get("active_compounds") or raw.get("supported_activities")),
        traditional_uses=traditional_uses,
        supported_activities=raw.get("supported_activities", []),
        evidence_level=evidence_level,
        preparation_methods=prep,
        usage_rules=usage,
        contraindications=[item.title for item in _safety_field(raw, "contraindications", source_ids).items],
        interactions=[item.title for item in _safety_field(raw, "interactions", source_ids).items],
        side_effects=[item.title for item in _safety_field(raw, "side_effects", source_ids).items],
        risk_groups=[item.title for item in _safety_field(raw, "risk_groups", source_ids).items],
        warnings=warnings,
        stop_use_signs=_clean_list(raw.get("stop_use_signs") or raw.get("stopUseSigns")),
        medical_attention_signs=medical_attention_signs(),
        availability=availability,
        availability_label=label,
        availability_reason=reason,
        recommendation_score=round(max(0.0, min(1.0, score)), 4),
        safety_status=safety_status,
        safety_reasons=safety_reasons,
        usage_status="available" if usage else "insufficient_data",
        graph_verified=graph_verified,
        provenance_valid=graph_verified,
        has_conflicting_claims=any(_safety_field(raw, key, source_ids).status == "conflicting" for key in ["contraindications", "interactions", "side_effects", "risk_groups"]),
        provenance=provenance,
        availability_info=availability_info,
        evidence=evidence_info,
        contraindication_status=_safety_field(raw, "contraindications", source_ids),
        interaction_status=_safety_field(raw, "interactions", source_ids),
        side_effect_status=_safety_field(raw, "side_effects", source_ids),
        risk_group_status=_safety_field(raw, "risk_groups", source_ids),
    )


def build_scored_candidates(raw_candidates: list[dict[str, Any]], symptoms: list[str], req: HerbalRecommendationRequest) -> list[HerbalCandidate]:
    scored = [score_candidate(raw, symptoms, req) for raw in raw_candidates]
    deduped, _ = deduplicate_herbal_candidates(scored)
    return deduped


def rank_scored_candidates(scored: list[HerbalCandidate], min_score: float, max_results: int) -> tuple[list[HerbalCandidate], list[dict[str, Any]]]:
    scored, _ = deduplicate_herbal_candidates(scored)
    candidates = [c for c in scored if c.safety_status != "excluded" and c.recommendation_score >= min_score]
    candidates, _ = deduplicate_herbal_candidates(candidates)
    excluded = [
        {"herb_id": c.herb_id, "canonical_key": c.canonical_key, "local_name": c.local_name, "safety_reasons": c.safety_reasons, "recommendation_score": c.recommendation_score}
        for c in scored if c.safety_status == "excluded" or c.recommendation_score < min_score
    ]
    candidates.sort(key=lambda c: (-c.recommendation_score, (c.scientific_name or c.local_name).casefold()))
    return candidates[:max_results], excluded


def rank_candidates(raw_candidates: list[dict[str, Any]], symptoms: list[str], req: HerbalRecommendationRequest, min_score: float, max_results: int) -> tuple[list[HerbalCandidate], list[dict[str, Any]]]:
    return rank_scored_candidates(build_scored_candidates(raw_candidates, symptoms, req), min_score, max_results)
