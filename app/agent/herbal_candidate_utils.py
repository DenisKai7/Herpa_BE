"""Canonical keys, deduplication, and response validation for herbal candidates."""

import re
from collections import OrderedDict
from typing import Iterable

from app.agent.herbal_verification import is_fully_verified_candidate
from app.models.herbal_recommendation import HerbalCandidate, HerbalRecommendationResponse, PreparationMethod, SafetyItem, UsageRule

BOTANICAL_AUTHOR_SUFFIXES = {
    "l", "linn", "linnaeus", "roxb", "roscoe", "willd", "nees", "kunth", "griff",
    "burm", "burm f", "r m sm", "valeton", "zijp", "ruiz", "pav",
}
VALID_EVIDENCE = {
    "traditional", "phytochemical_screening", "in_vitro", "in_vivo", "clinical",
    "systematic_review", "data_not_available", "insufficient_evidence",
}
VALID_AVAILABILITY = {"easy_to_find", "moderately_available", "hard_to_find", "seasonal", "restricted", "unknown"}
VALID_SAFETY = {"eligible", "conditional", "excluded"}
DEFAULT_CONDITIONAL_WARNING = (
    "Data keamanan spesifik belum lengkap. Penggunaan perlu berhati-hati, terutama pada kehamilan, "
    "menyusui, anak, lansia, penyakit kronis, alergi, dan pengguna obat rutin."
)


def merge_unique_strings(*collections: Iterable[str]) -> list[str]:
    seen: OrderedDict[str, str] = OrderedDict()
    for collection in collections:
        for value in collection or []:
            text = str(value).strip()
            if not text:
                continue
            key = " ".join(text.casefold().split())
            if key not in seen:
                seen[key] = text
    return list(seen.values())


def normalize_herb_name(value: str | None, *, strip_author: bool = False) -> str:
    if not value:
        return ""
    text = value.casefold()
    text = re.sub(r"[.,()]", " ", text)
    text = " ".join(text.split())
    if strip_author:
        parts = text.split()
        while len(parts) > 2 and parts[-1] in BOTANICAL_AUTHOR_SUFFIXES:
            parts.pop()
        text = " ".join(parts)
    return text


def build_canonical_herb_key_from_values(herb_id: str | None, scientific_name: str | None, local_name: str | None) -> str:
    scientific = normalize_herb_name(scientific_name, strip_author=True)
    if scientific:
        return f"scientific:{scientific}"
    local = normalize_herb_name(local_name)
    if local:
        return f"local:{local}"
    return f"id:{herb_id or 'unknown'}"


def build_canonical_herb_key(candidate: HerbalCandidate) -> str:
    return build_canonical_herb_key_from_values(candidate.herb_id, candidate.scientific_name, candidate.local_name)


def _merge_preparations(a: list[PreparationMethod], b: list[PreparationMethod]) -> list[PreparationMethod]:
    merged: OrderedDict[str, PreparationMethod] = OrderedDict()
    for item in [*a, *b]:
        key = item.method_id or f"{item.title}|{item.plant_part}|{'|'.join(item.steps)}"
        if key not in merged:
            merged[key] = item
    return list(merged.values())


def _merge_usage(a: list[UsageRule], b: list[UsageRule]) -> list[UsageRule]:
    merged: OrderedDict[str, UsageRule] = OrderedDict()
    for item in [*a, *b]:
        key = "|".join(str(v or "") for v in [item.form, item.amount_text, item.frequency_text, item.duration_text])
        if key not in merged:
            merged[key] = item
    return list(merged.values())


def _best_evidence(a: str, b: str) -> str:
    order = ["data_not_available", "insufficient_evidence", "traditional", "phytochemical_screening", "in_vitro", "in_vivo", "clinical", "systematic_review"]
    ia = order.index(a) if a in order else 1
    ib = order.index(b) if b in order else 1
    return a if ia >= ib else b


def _best_availability(primary: HerbalCandidate, duplicate: HerbalCandidate) -> tuple[str, str, str | None]:
    order = {"unknown": 0, "restricted": 1, "hard_to_find": 2, "seasonal": 3, "moderately_available": 4, "easy_to_find": 5}
    if order.get(duplicate.availability, 0) > order.get(primary.availability, 0):
        return duplicate.availability, duplicate.availability_label, duplicate.availability_reason
    return primary.availability, primary.availability_label, primary.availability_reason


def _merge_warnings(a: list[SafetyItem], b: list[SafetyItem]) -> list[SafetyItem]:
    merged: OrderedDict[str, SafetyItem] = OrderedDict()
    for item in [*a, *b]:
        key = item.safety_id or item.title
        if key not in merged:
            merged[key] = item
    return list(merged.values())


def merge_candidates(primary: HerbalCandidate, duplicate: HerbalCandidate) -> HerbalCandidate:
    availability, availability_label, availability_reason = _best_availability(primary, duplicate)
    primary.source_herb_ids = merge_unique_strings(primary.source_herb_ids or [primary.herb_id], duplicate.source_herb_ids or [duplicate.herb_id])
    primary.matched_symptoms = merge_unique_strings(primary.matched_symptoms, duplicate.matched_symptoms)
    primary.unmatched_symptoms = merge_unique_strings(primary.unmatched_symptoms, duplicate.unmatched_symptoms)
    primary.traditional_uses = merge_unique_strings(primary.traditional_uses, duplicate.traditional_uses)
    primary.supported_activities = merge_unique_strings(primary.supported_activities, duplicate.supported_activities)
    primary.contraindications = merge_unique_strings(primary.contraindications, duplicate.contraindications)
    primary.interactions = merge_unique_strings(primary.interactions, duplicate.interactions)
    primary.side_effects = merge_unique_strings(primary.side_effects, duplicate.side_effects)
    primary.risk_groups = merge_unique_strings(primary.risk_groups, duplicate.risk_groups)
    primary.warnings = _merge_warnings(primary.warnings, duplicate.warnings)
    primary.stop_use_signs = merge_unique_strings(primary.stop_use_signs, duplicate.stop_use_signs)
    primary.medical_attention_signs = merge_unique_strings(primary.medical_attention_signs, duplicate.medical_attention_signs)
    primary.aliases = merge_unique_strings(primary.aliases, duplicate.aliases)
    primary.preparation_methods = _merge_preparations(primary.preparation_methods, duplicate.preparation_methods)
    primary.usage_rules = _merge_usage(primary.usage_rules, duplicate.usage_rules)
    primary.evidence_level = _best_evidence(primary.evidence_level, duplicate.evidence_level)
    primary.availability = availability  # type: ignore[assignment]
    primary.availability_label = availability_label
    primary.availability_reason = availability_reason
    primary.recommendation_score = max(primary.recommendation_score, duplicate.recommendation_score)
    if primary.safety_status == "eligible" and duplicate.safety_status == "conditional":
        primary.safety_status = "conditional"
    primary.safety_reasons = merge_unique_strings(primary.safety_reasons, duplicate.safety_reasons)
    if primary.explanation is None:
        primary.explanation = duplicate.explanation
    return primary


def deduplicate_herbal_candidates(candidates: list[HerbalCandidate]) -> tuple[list[HerbalCandidate], list[str]]:
    by_key: OrderedDict[str, HerbalCandidate] = OrderedDict()
    duplicate_keys: list[str] = []
    for candidate in candidates:
        candidate.canonical_key = candidate.canonical_key or build_canonical_herb_key(candidate)
        if candidate.canonical_key not in by_key:
            if not candidate.source_herb_ids:
                candidate.source_herb_ids = [candidate.herb_id]
            by_key[candidate.canonical_key] = candidate
        else:
            duplicate_keys.append(candidate.canonical_key)
            by_key[candidate.canonical_key] = merge_candidates(by_key[candidate.canonical_key], candidate)
    return list(by_key.values()), duplicate_keys


def _sanitize_candidate(candidate: HerbalCandidate) -> HerbalCandidate:
    if not candidate.herb_id:
        candidate.herb_id = candidate.canonical_key or build_canonical_herb_key(candidate)
    if not candidate.local_name or candidate.local_name == "data_not_available":
        candidate.local_name = candidate.scientific_name or "Tanaman belum teridentifikasi"
    candidate.canonical_key = build_canonical_herb_key(candidate)
    candidate.recommendation_score = max(0.0, min(1.0, candidate.recommendation_score))
    if candidate.evidence_level not in VALID_EVIDENCE:
        candidate.evidence_level = "insufficient_evidence"
    if candidate.availability not in VALID_AVAILABILITY:
        candidate.availability = "unknown"  # type: ignore[assignment]
        candidate.availability_label = "Ketersediaan belum diketahui"
    if candidate.safety_status not in VALID_SAFETY:
        candidate.safety_status = "conditional"  # type: ignore[assignment]
    return candidate


def validate_herbal_recommendation_response(response: HerbalRecommendationResponse) -> HerbalRecommendationResponse:
    sanitized = [_sanitize_candidate(candidate) for candidate in response.recommendations]
    deduped, duplicates = deduplicate_herbal_candidates(sanitized)
    scientific_seen: set[str] = set()
    unique_by_scientific: list[HerbalCandidate] = []
    scientific_duplicates = 0
    for candidate in deduped:
        sci = normalize_herb_name(candidate.scientific_name, strip_author=True)
        if sci and sci in scientific_seen:
            scientific_duplicates += 1
            continue
        if sci:
            scientific_seen.add(sci)
        if candidate.safety_status == "excluded" or not is_fully_verified_candidate(candidate):
            response.excluded_candidates.append({
                "herb_id": candidate.herb_id,
                "canonical_key": candidate.canonical_key,
                "local_name": candidate.local_name,
                "safety_reasons": candidate.safety_reasons,
                "missing_fields": candidate.verification_coverage.missing_fields if candidate.verification_coverage else [],
            })
        else:
            unique_by_scientific.append(candidate)
    response.recommendations = unique_by_scientific
    if not response.recommendations and response.status == "completed":
        response.status = "no_fully_verified_candidate"  # type: ignore[assignment]
        response.medical_attention_message = "Knowledge graph belum memiliki kandidat dengan data penggunaan dan keamanan yang lengkap untuk keluhan ini."
    response.total_candidates_eligible = len(response.recommendations)
    response.total_candidates_excluded = len(response.excluded_candidates)
    response.metadata["response_duplicate_count"] = len(duplicates) + scientific_duplicates
    return response
