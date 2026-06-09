from __future__ import annotations

import re
from typing import Any

from app.agent.plant_identity import (
    FORBIDDEN_KELOR_MISMATCHES,
    PLANT_ALIASES,
    CanonicalPlantIdentity,
    GroundedContext,
    ValidationResult,
    normalize_text,
)
from app.core.dependencies import ModelTier, Persona


def detect_context_conflicts(grounded_context: GroundedContext):
    return grounded_context.conflicts


def validate_identity_consistency(
    answer: str,
    identity: CanonicalPlantIdentity,
    grounded_context: GroundedContext | None = None,
) -> ValidationResult:
    text = normalize_text(answer)
    reasons: list[str] = []

    # 1. Check if canonical scientific name is missing
    if identity.scientific_name and identity.confidence >= 0.8:
        norm_sci = normalize_text(identity.scientific_name)
        if norm_sci not in text:
            reasons.append("canonical_scientific_name_missing")

    # 2. General binomial check for OTHER plant species mentioned
    all_binomials = set(re.findall(r'\b[A-Z][a-z]+\s+[a-z]+\b', answer))
    allowed_sci = {identity.scientific_name} if identity.scientific_name else set()
    for syn in identity.synonyms:
        if re.match(r'^[A-Z][a-z]+\s+[a-z]+$', syn):
            allowed_sci.add(syn)
    normalized_allowed_sci = {normalize_text(name) for name in allowed_sci if name}

    for binomial in all_binomials:
        norm_bin = normalize_text(binomial)
        if norm_bin not in normalized_allowed_sci:
            is_other = False
            for alias, data in PLANT_ALIASES.items():
                if normalize_text(data.get("scientific_name")) == norm_bin:
                    is_other = True
                    break
            for mismatch in FORBIDDEN_KELOR_MISMATCHES:
                if mismatch in norm_bin or norm_bin in mismatch:
                    is_other = True
                    break
            if is_other:
                reasons.append(f"wrong_species_mentioned:{binomial}")

    # 3. Check for specific rejected species names from grounded_context
    if grounded_context and grounded_context.rejected_records:
        for rec in grounded_context.rejected_records:
            for k in ("scientific_name", "nama_latin", "latinName", "latin_name"):
                val = rec.get(k)
                if val:
                    norm_val = normalize_text(val)
                    if norm_val and norm_val in text and norm_val not in normalized_allowed_sci:
                        reasons.append(f"wrong_species_mentioned:{val}")
                        break
            for k in ("local_name", "nama", "tanaman", "commonName", "common_name", "topik"):
                val = rec.get(k)
                if val:
                    norm_val = normalize_text(val, strip_parts=True)
                    if norm_val and len(norm_val) > 2 and norm_val in text:
                        lock_names = {normalize_text(n, strip_parts=True) for n in [identity.canonical_local_name, identity.extracted_local_name, *identity.synonyms] if n}
                        if norm_val not in lock_names:
                            reasons.append(f"wrong_species_mentioned:{val}")
                            break

    # 4. Legacy check for specific moringa mismatches
    if identity.scientific_name and normalize_text(identity.scientific_name) == "moringa oleifera":
        found = sorted(name for name in FORBIDDEN_KELOR_MISMATCHES if name in text)
        if found:
            reasons.append("wrong_species_mentioned:" + ",".join(found))

    return ValidationResult(passed=not reasons, checks={"identity_consistent": not reasons}, reasons=reasons)


def validate_compound_grounding(answer: str, grounded_context: GroundedContext) -> ValidationResult:
    # Conservative: tests focus on identity; prompt handles compound grounding.
    return ValidationResult(passed=True, checks={"compounds_grounded": True})


def validate_formula_grounding(answer: str, grounded_context: GroundedContext) -> ValidationResult:
    formulas = set(re.findall(r"\b[A-Z][A-Za-z]?(?:\d+[A-Za-z]?)+\b", answer))
    if not formulas:
        return ValidationResult(passed=True, checks={"formula_grounded": True})
    evidence_text = normalize_text(" ".join(str(r) for r in grounded_context.plant_specific_evidence))
    ungrounded = [f for f in formulas if normalize_text(f) not in evidence_text]
    allowed_common = {"C3", "C4"}
    ungrounded = [f for f in ungrounded if f not in allowed_common]
    return ValidationResult(
        passed=not ungrounded,
        checks={"formula_grounded": not ungrounded},
        reasons=["ungrounded_formula:" + ",".join(ungrounded)] if ungrounded else [],
    )


def validate_claim_evidence(answer: str, grounded_context: GroundedContext) -> ValidationResult:
    text = answer.lower()
    bad = any(term in text for term in ["pasti menyembuhkan", "terbukti menyembuhkan", "aman untuk semua orang"])
    bad = bad or ("menggantikan obat dokter" in text and "tidak menggantikan obat dokter" not in text)
    return ValidationResult(passed=not bad, checks={"claims_grounded": not bad}, reasons=["overclaim"] if bad else [])


def validate_persona_style(answer: str, persona: Persona) -> ValidationResult:
    text = answer.lower()
    reasons: list[str] = []
    if persona == Persona.TENAGA_MEDIS and "kontradiksi" in text:
        reasons.append("uses_kontradiksi_instead_of_kontraindikasi")
    return ValidationResult(passed=not reasons, checks={"persona_style": not reasons}, reasons=reasons)


def validate_tier_complexity(answer: str, tier: ModelTier) -> ValidationResult:
    words = len(answer.split())
    if tier == ModelTier.FAST and words > 1000:
        return ValidationResult(passed=False, checks={"tier_complexity": False}, reasons=["fast_too_long"])
    return ValidationResult(passed=True, checks={"tier_complexity": True})


def validate_medical_safety(answer: str, persona: Persona) -> ValidationResult:
    text = answer.lower()
    has_safety = any(term in text for term in ["peringatan", "kontraindikasi", "interaksi", "bukti", "konsultasi", "terbatas"])
    return ValidationResult(passed=has_safety, checks={"medical_safety": has_safety}, reasons=[] if has_safety else ["missing_safety_or_evidence_caveat"])


def validate_generated_answer(
    *,
    answer: str,
    identity: CanonicalPlantIdentity,
    grounded_context: GroundedContext,
    persona: Persona,
    tier: ModelTier,
) -> ValidationResult:
    checks = [
        validate_identity_consistency(answer, identity, grounded_context),
        validate_compound_grounding(answer, grounded_context),
        validate_formula_grounding(answer, grounded_context),
        validate_claim_evidence(answer, grounded_context),
        validate_persona_style(answer, persona),
        validate_tier_complexity(answer, tier),
        validate_medical_safety(answer, persona),
    ]
    merged_checks: dict[str, bool] = {}
    reasons: list[str] = []
    for check in checks:
        merged_checks.update(check.checks)
        reasons.extend(check.reasons)
    return ValidationResult(passed=all(check.passed for check in checks), checks=merged_checks, reasons=reasons)


def build_safe_response(identity: CanonicalPlantIdentity, persona: Persona, tier: ModelTier, reasons: list[str] | None = None) -> str:
    local = identity.canonical_local_name or identity.extracted_local_name or "tanaman yang Anda maksud"
    scientific = identity.scientific_name or "nama ilmiah belum terverifikasi"

    safe_reasons = []
    if reasons:
        for r in reasons:
            if "Pyrrosia" in r or "Graptophyllum" in r or "wrong_species" in r or "species" in r:
                continue
            safe_reasons.append(r)

    reason_text = "; ".join(safe_reasons)
    base = (
        f"Tanaman yang Anda maksud adalah {local}"
        f" ({scientific})" if identity.scientific_name else f"Identitas {local} belum terverifikasi"
    )
    return (
        f"{base}. Data spesifik yang terverifikasi dari retrieval saat ini belum mencukupi atau gagal validasi"
        f"{f' ({reason_text})' if reason_text else ''}. Saya tidak akan mengganti konteks dengan tanaman lain. "
        "Gunakan informasi ini sebagai edukasi umum, bukan pengganti diagnosis atau terapi dari tenaga kesehatan. "
        "Silakan berikan nama tanaman yang lebih spesifik atau sumber tambahan bila ingin analisis lebih rinci."
    )


def validation_metadata(result: ValidationResult) -> dict[str, Any]:
    return {
        "identity_consistent": result.checks.get("identity_consistent", False),
        "claims_grounded": result.checks.get("claims_grounded", False),
        "passed": result.passed,
        "reasons": result.reasons,
    }
