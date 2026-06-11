"""Post-LLM grounding checks for graph-verified herbal explanations."""

import re
from typing import Any

from app.models.herbal_recommendation import HerbalCandidate


def _numbers(text: str) -> set[str]:
    return set(re.findall(r"\d+(?:[,.]\d+)?", text or ""))


def _lower_tokens(values: list[str]) -> set[str]:
    return {str(value).casefold().strip() for value in values if str(value).strip()}


def validate_candidate_ids(explanations: dict[str, str], candidates: list[HerbalCandidate]) -> bool:
    allowed = {candidate.canonical_key for candidate in candidates}
    return set(explanations).issubset(allowed)


def validate_numeric_fidelity(text: str, candidate: HerbalCandidate) -> bool:
    source_text = " ".join(
        [
            *(step for method in candidate.preparation_methods for step in method.steps),
            *(method.title for method in candidate.preparation_methods),
            *(rule.amount_text or "" for rule in candidate.usage_rules),
            *(rule.frequency_text or "" for rule in candidate.usage_rules),
            *(rule.duration_text or "" for rule in candidate.usage_rules),
        ]
    )
    return _numbers(text).issubset(_numbers(source_text))


def validate_ingredient_fidelity(text: str, candidate: HerbalCandidate) -> bool:
    ingredients = _lower_tokens([item.name for method in candidate.preparation_methods for item in method.ingredients])
    if not ingredients:
        return True
    lowered = text.casefold()
    common_added_ingredients = {"madu", "gula", "garam", "jeruk", "susu"}
    added = [item for item in common_added_ingredients if item in lowered and item not in ingredients]
    return not added


def validate_preparation_step_fidelity(text: str, candidate: HerbalCandidate) -> bool:
    if not candidate.preparation_methods:
        return "rebus" not in text.casefold() and "seduh" not in text.casefold()
    return True


def validate_usage_rule_fidelity(text: str, candidate: HerbalCandidate) -> bool:
    if not candidate.usage_rules:
        return not any(term in text.casefold() for term in ["sehari", "dosis", "kali sehari", "sendok"])
    return True


def validate_safety_fidelity(text: str, candidate: HerbalCandidate) -> bool:
    warning_terms = [item.title for item in candidate.warnings] + [item.description for item in candidate.warnings]
    source_terms = " ".join(candidate.contraindications + candidate.interactions + candidate.side_effects + candidate.risk_groups + warning_terms).casefold()
    lowered = text.casefold()
    sensitive_terms = ["hamil", "menyusui", "antikoagulan", "diabetes", "ginjal", "hati"]
    return all(term not in lowered or term in source_terms for term in sensitive_terms)


def validate_source_coverage(candidate: HerbalCandidate) -> bool:
    return bool(candidate.provenance and candidate.provenance.source_ids and candidate.provenance.coverage_score == 1.0)


def validate_no_new_claims(text: str, candidate: HerbalCandidate) -> bool:
    return all([
        validate_numeric_fidelity(text, candidate),
        validate_ingredient_fidelity(text, candidate),
        validate_preparation_step_fidelity(text, candidate),
        validate_usage_rule_fidelity(text, candidate),
        validate_safety_fidelity(text, candidate),
    ])


def deterministic_graph_explanation(candidate: HerbalCandidate) -> str:
    uses = ", ".join(candidate.traditional_uses[:3]) or "keluhan yang cocok pada knowledge graph"
    return (
        f"{candidate.local_name} dipilih karena memiliki kecocokan graph untuk {uses}. "
        "Seluruh informasi yang ditampilkan telah memiliki provenance pada knowledge graph."
    )


def validate_grounded_explanations(explanations: dict[str, str], candidates: list[HerbalCandidate]) -> tuple[dict[str, str], list[dict[str, Any]]]:
    violations: list[dict[str, Any]] = []
    valid: dict[str, str] = {}
    by_id = {candidate.canonical_key: candidate for candidate in candidates}
    if not validate_candidate_ids(explanations, candidates):
        unknown = sorted(set(explanations) - set(by_id))
        violations.append({"type": "unknown_candidate_id", "ids": unknown})
    for candidate in candidates:
        text = explanations.get(candidate.canonical_key) or ""
        if not text or not validate_source_coverage(candidate) or not validate_no_new_claims(text, candidate):
            violations.append({"type": "llm_grounding_rejected", "candidate_id": candidate.canonical_key})
            valid[candidate.canonical_key] = deterministic_graph_explanation(candidate)
        else:
            valid[candidate.canonical_key] = text
    return valid, violations
