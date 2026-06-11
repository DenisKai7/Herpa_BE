"""Graph provenance coverage, dual-verification scoring, and deterministic safety rules."""

import logging
from typing import Any
from app.models.herbal_recommendation import (
    FieldVerification,
    GraphProvenance,
    HerbVerificationCoverage,
    HerbalCandidate,
    PreparationMethod,
    UsageRule,
    VerifiedSafetyField,
    VerificationSource,
    GRAPH_COVERAGE_WEIGHTS,
    GENERAL_SAFETY_WARNING,
)

logger = logging.getLogger(__name__)

REQUIRED_RECOMMENDATION_FIELDS = {
    "identity",
    "therapeutic_use",
    "preparation",
    "usage_rule",
    "contraindication_status",
    "interaction_status",
    "side_effect_status",
    "risk_group_status",
    "warning",
    "availability",
    "provenance",
}
REQUIRED_FIELDS = list(REQUIRED_RECOMMENDATION_FIELDS)


def _has_sources(values: list[str] | None) -> bool:
    return bool(values and all(str(v).strip() for v in values))


def _safety_verified(field: VerifiedSafetyField) -> bool:
    if field.status not in {"known_issue", "no_known_issue_within_source_scope"}:
        return False
    if not _has_sources(field.source_ids):
        return False
    if field.status == "known_issue":
        return bool(field.items) and all(_has_sources(item.source_ids) for item in field.items)
    return True


def _preparation_verified(methods: list[PreparationMethod]) -> bool:
    if not methods:
        return False
    for method in methods:
        if method.verification_status != "verified":
            return False
        if not all([method.method_id, method.title, method.plant_part, method.dosage_form]):
            return False
        if not method.steps or not method.ingredients or not method.suitable_symptoms:
            return False
        if not _has_sources(method.source_ids):
            return False
        if any(not ingredient.name or not _has_sources(ingredient.source_ids) for ingredient in method.ingredients):
            return False
    return True


def _usage_verified(rules: list[UsageRule]) -> bool:
    if not rules:
        return False
    for rule in rules:
        if rule.verification_status != "verified":
            return False
        if not all([rule.usage_rule_id, rule.amount_text, rule.frequency_text, rule.duration_text, rule.evidence_level]):
            return False
        if not _has_sources(rule.source_ids):
            return False
    return True


def _warnings_verified(candidate: HerbalCandidate) -> bool:
    return bool(candidate.warnings) and all(
        item.safety_id and item.title and item.description and item.action_text and _has_sources(item.source_ids)
        for item in candidate.warnings
    ) and bool(candidate.stop_use_signs)


def _provenance_verified(candidate: HerbalCandidate) -> bool:
    if not candidate.provenance:
        return False
    if not candidate.provenance.graph_verified:
        return False
    if not _has_sources(candidate.provenance.source_ids):
        return False
    if not candidate.provenance.graph_node_ids:
        return False
    source_ids = set(candidate.provenance.source_ids)
    if candidate.provenance.sources:
        detailed_ids = {source.source_id for source in candidate.provenance.sources if source.source_id and source.title}
        if not source_ids.issubset(detailed_ids):
            return False
    return True


def build_safe_general_preparation(
    herb: HerbalCandidate,
) -> FieldVerification:
    return FieldVerification(
        field_name="preparation_method",
        value={
            "title": "Panduan pengolahan umum",
            "steps": [
                "Pastikan identitas tanaman benar.",
                "Gunakan bagian tanaman yang tercatat pada knowledge graph.",
                "Cuci bahan dengan air mengalir.",
                "Gunakan peralatan yang bersih.",
                "Hindari menambahkan bahan lain yang belum diketahui keamanannya.",
            ],
        },
        verification_source=VerificationSource.MODEL_ASSISTED,
        model_confidence=None,
        model_critic_passed=False,
        safety_critical=False,
        warnings=[
            "Takaran, suhu, dan durasi spesifik belum mempunyai sumber terverifikasi."
        ],
    )


def build_safe_general_usage(
    herb: HerbalCandidate,
) -> FieldVerification:
    return FieldVerification(
        field_name="usage_rule",
        value={
            "title": "Panduan penggunaan umum",
            "instructions": [
                "Gunakan hanya setelah identitas tanaman dipastikan.",
                "Jangan menggunakan dalam jumlah berlebihan.",
                "Jangan digunakan sebagai pengganti obat resep.",
                "Hentikan penggunaan jika muncul reaksi yang tidak diinginkan."
            ],
            "specific_dosage_available": False,
        },
        verification_source=VerificationSource.MODEL_ASSISTED,
        model_confidence=None,
        model_critic_passed=False,
        safety_critical=True,
        warnings=[
            "Aturan pakai spesifik belum mempunyai sumber terverifikasi. Jangan menentukan dosis sendiri, terutama untuk anak, kehamilan, lansia, penyakit kronis, atau pengguna obat rutin."
        ],
    )


def build_safe_general_warnings(
    herb: HerbalCandidate,
) -> list[FieldVerification]:
    return [
        FieldVerification(
            field_name="general_safety_warning",
            value=GENERAL_SAFETY_WARNING,
            verification_source=VerificationSource.MODEL_ASSISTED,
            model_confidence=None,
            model_critic_passed=False,
            safety_critical=True,
            warnings=["Peringatan umum keselamatan"],
        )
    ]


def calculate_dual_verification(candidate: HerbalCandidate) -> None:
    """Calculates coverage scores and overall verification status."""
    # Perform legacy checks first to maintain attributes
    source_ids: list[str] = []
    if candidate.provenance:
        source_ids.extend(candidate.provenance.source_ids)
    if candidate.availability_info:
        source_ids.extend(candidate.availability_info.source_ids)
    if candidate.evidence:
        source_ids.extend(candidate.evidence.source_ids)
    for method in candidate.preparation_methods:
        source_ids.extend(method.source_ids)
        for ingredient in method.ingredients:
            source_ids.extend(ingredient.source_ids)
    for rule in candidate.usage_rules:
        source_ids.extend(rule.source_ids)
    for field in [candidate.contraindication_status, candidate.interaction_status, candidate.side_effect_status, candidate.risk_group_status]:
        source_ids.extend(field.source_ids)
        for item in field.items:
            source_ids.extend(item.source_ids)
    for warning in candidate.warnings:
        source_ids.extend(warning.source_ids)
    source_ids = list(dict.fromkeys(source_ids))

    checks = {
        "identity": bool(candidate.herb_id and candidate.local_name and candidate.scientific_name),
        "symptom_relevance": bool(candidate.matched_symptoms),
        "therapeutic_use": bool(candidate.traditional_uses and candidate.evidence and _has_sources(candidate.evidence.source_ids)),
        "active_compounds": bool(candidate.active_compounds),
        "preparation": _preparation_verified(candidate.preparation_methods),
        "usage_rule": _usage_verified(candidate.usage_rules),
        "contraindication_status": _safety_verified(candidate.contraindication_status),
        "interaction_status": _safety_verified(candidate.interaction_status),
        "side_effect_status": _safety_verified(candidate.side_effect_status),
        "risk_group_status": _safety_verified(candidate.risk_group_status),
        "warning": _warnings_verified(candidate),
        "availability": bool(candidate.availability_info and candidate.availability != "unknown" and _has_sources(candidate.availability_info.source_ids)),
        "provenance": _provenance_verified(candidate),
    }

    # Legacy attributes compatibility
    missing_fields_legacy = [key for key in REQUIRED_FIELDS if not checks.get(key, False)]
    verified_count_legacy = len(REQUIRED_FIELDS) - len(missing_fields_legacy)
    candidate.verification_coverage = HerbVerificationCoverage(
        herb_id=candidate.herb_id,
        identity_verified=checks["identity"],
        therapeutic_use_verified=checks["therapeutic_use"],
        preparation_verified=checks["preparation"],
        usage_rule_verified=checks["usage_rule"],
        contraindication_verified=checks["contraindication_status"],
        interaction_verified=checks["interaction_status"],
        side_effect_verified=checks["side_effect_status"],
        risk_group_verified=checks["risk_group_status"],
        warning_verified=checks["warning"],
        availability_verified=checks["availability"],
        provenance_verified=checks["provenance"],
        verified_field_count=verified_count_legacy,
        required_field_count=len(REQUIRED_FIELDS),
        coverage_score=round(verified_count_legacy / len(REQUIRED_FIELDS), 4),
        source_ids=source_ids,
        missing_fields=missing_fields_legacy,
    )
    if not candidate.provenance:
        candidate.provenance = GraphProvenance()
    candidate.provenance.coverage_score = candidate.verification_coverage.coverage_score
    candidate.provenance.source_ids = source_ids
    candidate.provenance_valid = bool(candidate.provenance and candidate.provenance.graph_verified and candidate.provenance.coverage_score >= 1.0)
    candidate.graph_verified = bool(candidate.provenance and candidate.provenance.graph_verified)

    # Let's count weights for graph_coverage_score
    # Weights schema mapping:
    # identity                 0.15
    # symptom relevance        0.15
    # therapeutic use          0.10
    # active compounds         0.10
    # preparation              0.10
    # usage rule               0.10
    # contraindication         0.08
    # interaction              0.08
    # side effects             0.05
    # risk groups              0.04
    # availability             0.03
    # provenance               0.02
    graph_coverage = 0.0
    field_checks = {
        "identity": checks["identity"],
        "symptom_relevance": checks["symptom_relevance"],
        "therapeutic_use": checks["therapeutic_use"],
        "active_compounds": checks["active_compounds"],
        "preparation": checks["preparation"],
        "usage_rule": checks["usage_rule"],
        "contraindication": checks["contraindication_status"],
        "interaction": checks["interaction_status"],
        "side_effects": checks["side_effect_status"],
        "risk_groups": checks["risk_group_status"],
        "availability": checks["availability"],
        "provenance": checks["provenance"],
    }

    for key, weight in GRAPH_COVERAGE_WEIGHTS.items():
        if field_checks.get(key, False):
            graph_coverage += weight

    candidate.graph_coverage_score = round(graph_coverage, 4)

    # Calculate model_assisted_coverage_score:
    # Start with graph_coverage_score (only graph data)
    # Then add weight of any non-critical field completed by model that passed critic
    # Note: Model-assisted fields do NOT increase the graph score.
    model_coverage = graph_coverage
    field_verif_map = {fv.field_name: fv for fv in candidate.field_verifications}

    # We will map preparation, usage_rule, availability model assistance
    # (availability is non-critical, preparation can be model_assisted for general,
    # but specific usage rules and safety data are safety-critical and cannot be model_assisted)

    if not checks["preparation"]:
        fv = field_verif_map.get("preparation_method")
        if fv and fv.verification_source == VerificationSource.MODEL_ASSISTED and fv.model_critic_passed:
            model_coverage += GRAPH_COVERAGE_WEIGHTS["preparation"]

    if not checks["availability"]:
        fv = field_verif_map.get("availability")
        if fv and fv.verification_source == VerificationSource.MODEL_ASSISTED and fv.model_critic_passed:
            model_coverage += GRAPH_COVERAGE_WEIGHTS["availability"]

    candidate.model_assisted_coverage_score = round(model_coverage, 4)

    # Calculate safety_data_status
    # If Neo4j does not have contraindications, interactions, side_effects, or risk_groups
    # (i.e. they are missing), safety_data_status = incomplete or missing.
    safety_keys = ["contraindication_status", "interaction_status", "side_effect_status", "risk_group_status"]
    safety_verified_count = sum(1 for k in safety_keys if checks[k])
    if safety_verified_count == 4:
        candidate.safety_data_status = "complete"
    elif safety_verified_count == 0:
        candidate.safety_data_status = "missing"
    else:
        candidate.safety_data_status = "incomplete"

    # Set overall_verification_status:
    # 1. fully_graph_verified: All fields verified in Graph (graph_coverage_score == 1.0 and safety complete)
    # 2. graph_and_model_verified: Identitas, kegunaan, senyawa, safety berasal dari graph. Model hanya merangkum/menyederhanakan.
    # 3. model_assisted_limited: Identitas + gejala dari graph, but prep/availability model-assisted. No fake safety/usage.
    # 4. insufficient_data: Identitas/gejala tidak cukup kuat, or key data missing.

    field_verif_names = {fv.field_name for fv in candidate.field_verifications}
    covered = {
        "identity": checks["identity"],
        "therapeutic_use": checks["therapeutic_use"],
        "preparation": checks["preparation"] or ("preparation_method" in field_verif_names),
        "usage_rule": checks["usage_rule"] or ("usage_rule" in field_verif_names),
        "contraindication_status": checks["contraindication_status"] or ("contraindication" in field_verif_names),
        "interaction_status": checks["interaction_status"] or ("interaction" in field_verif_names),
        "side_effect_status": checks["side_effect_status"] or ("side_effects" in field_verif_names),
        "risk_group_status": checks["risk_group_status"] or ("risk_groups" in field_verif_names),
        "warning": checks["warning"] or ("general_safety_warning" in field_verif_names),
        "availability": checks["availability"] or ("availability" in field_verif_names),
        "provenance": checks["provenance"] or ("provenance" in field_verif_names),
    }

    # Core requirements for graph_and_model_verified:
    # identity, symptom_relevance, therapeutic_use, active_compounds, and safety (contra, interaction, side, risk) must be graph-verified.
    core_graph_verified = (
        checks["identity"] and
        checks["symptom_relevance"] and
        checks["therapeutic_use"] and
        checks["active_compounds"] and
        checks["contraindication_status"] and
        checks["interaction_status"] and
        checks["side_effect_status"] and
        checks["risk_group_status"]
    )

    # Check if model has completed any non-critical field
    has_model_assisted = any(
        fv.verification_source == VerificationSource.MODEL_ASSISTED
        for fv in candidate.field_verifications
        if fv.field_name not in {"general_safety_warning", "usage_rule"}
    )

    if not all(covered.values()):
        candidate.overall_verification_status = "insufficient_data"
    elif has_model_assisted:
        candidate.overall_verification_status = "model_assisted_limited"
    elif graph_coverage >= 0.99 and candidate.safety_data_status == "complete":
        candidate.overall_verification_status = "fully_graph_verified"
    elif core_graph_verified:
        candidate.overall_verification_status = "graph_and_model_verified"
    elif checks["identity"] and checks["symptom_relevance"]:
        candidate.overall_verification_status = "model_assisted_limited"
    else:
        candidate.overall_verification_status = "insufficient_data"


def is_fully_verified_candidate(candidate: HerbalCandidate) -> bool:
    """Backward compatible gate for candidate qualification."""
    # Ensure verification scores are computed
    if candidate.graph_coverage_score == 0.0:
        calculate_dual_verification(candidate)

    # In the new dual verification model, candidates can be:
    # fully_graph_verified, graph_and_model_verified, model_assisted_limited.
    # We reject insufficient_data.
    # And candidate must have safety_status not excluded.
    return (
        candidate.overall_verification_status in {"fully_graph_verified", "graph_and_model_verified", "model_assisted_limited"}
        and candidate.safety_status != "excluded"
    )


def calculate_herb_verification_coverage(candidate: HerbalCandidate) -> HerbVerificationCoverage:
    """Legacy compatibility coverage calculator."""
    calculate_dual_verification(candidate)
    return candidate.verification_coverage


def rejection_reason_summary(candidates: list[HerbalCandidate]) -> dict[str, int]:
    from collections import Counter
    counter: Counter[str] = Counter()
    for candidate in candidates:
        if candidate.verification_coverage:
            for field in candidate.verification_coverage.missing_fields:
                counter[f"missing_{field}"] += 1
        if candidate.has_conflicting_claims:
            counter["conflicting"] += 1
        if not candidate.provenance or not candidate.provenance.source_ids:
            counter["missing_source"] += 1
    return dict(counter)
