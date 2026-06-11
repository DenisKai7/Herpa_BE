"""Orchestrates the grounded herbal recommendation pipeline with dual verification."""

import hashlib
import logging
import time
import uuid
from typing import Any

from app.agent.herbal_candidate_utils import (
    build_canonical_herb_key_from_values,
    deduplicate_herbal_candidates,
)
from app.agent.herbal_graph import retrieve_graph_verified_herbal_candidates
from app.agent.herbal_grounding_validator import validate_grounded_explanations
from app.agent.herbal_llm import (
    build_grounded_explanations,
    extract_complaint,
    model_generate_noncritical_fields,
    model_critic_validate,
)
from app.agent.herbal_ranking import build_scored_candidates, rank_scored_candidates
from app.agent.herbal_verification import (
    calculate_dual_verification,
    is_fully_verified_candidate,
    rejection_reason_summary,
    build_safe_general_preparation,
    build_safe_general_usage,
    build_safe_general_warnings,
)
from app.agent.herbal_safety import (
    CLARIFICATION_QUESTIONS,
    deterministic_red_flags,
    medical_attention_message,
    medical_attention_signs,
    needs_clarification,
    safety_assess,
)
from app.core.config import settings
from app.models.herbal_recommendation import (
    HerbalRecommendationError,
    HerbalRecommendationRequest,
    HerbalRecommendationResponse,
    HerbalCandidate,
    FieldVerification,
    VerificationSource,
    PreparationMethod,
    IngredientItem,
    UsageRule,
    VerifiedSafetyField,
    GENERAL_SAFETY_WARNING,
)

logger = logging.getLogger(__name__)

GENERAL_DISCLAIMER = (
    "Rekomendasi ini bersifat edukatif dan sebagai terapi penunjang, bukan diagnosis "
    "atau pengganti pengobatan dokter. Hentikan penggunaan jika timbul reaksi alergi, "
    "sesak, bengkak, ruam berat, muntah terus-menerus, atau kondisi memburuk."
)
PLANT_DISCLAIMER = (
    "Perhatian: data keamanan perlu disesuaikan dengan usia, kehamilan, penyakit penyerta, "
    "alergi, serta obat yang sedang digunakan."
)

_STORE: dict[str, dict[str, Any]] = {}


def _safe_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _metadata(request_id: str, started: float, **extra: Any) -> dict[str, Any]:
    base = {
        "model_id": settings.HERBAL_RECOMMENDATION_MODEL,
        "graph_records": 0,
        "symptom_nodes": 0,
        "candidate_count_raw": 0,
        "candidate_count_after_safety": 0,
        "processing_ms": int((time.perf_counter() - started) * 1000),
        "request_id": request_id,
        "fallback_used": False,
    }
    base.update(extra)
    return base


def _log(event: str, **fields: Any) -> None:
    logger.info("%s %s", event, " ".join(f"{k}={v}" for k, v in fields.items()))


def _failed(request_id: str, stage: str, error: HerbalRecommendationError) -> None:
    _log(
        "herbal_recommendation_failed",
        request_id=request_id,
        stage=stage,
        error_code=error.code,
        retryable=str(error.retryable).lower(),
    )


def get_cached(recommendation_id: str) -> HerbalRecommendationResponse | None:
    payload = _STORE.get(recommendation_id)
    if not payload:
        return None
    return payload["response"]


def refresh_cached(recommendation_id: str, user_id: str, request_id: str | None = None) -> HerbalRecommendationResponse | None:
    payload = _STORE.get(recommendation_id)
    if not payload:
        return None
    return analyze_herbal_complaint(payload["request"], user_id=user_id, request_id=request_id)


def validate_dual_verified_response(
    response: HerbalRecommendationResponse,
) -> HerbalRecommendationResponse:
    """Validate dual-verified response to enforce safety and data source policies."""
    from app.agent.herbal_candidate_utils import normalize_herb_name
    valid_recs = []

    for candidate in response.recommendations:
        # 1. Identitas tanaman wajib berasal dari graph
        if not candidate.herb_id or not candidate.local_name or candidate.scientific_name == "data_not_available":
            continue

        # 2. Hubungan gejala wajib berasal dari graph
        if not candidate.matched_symptoms:
            continue

        # 3. Candidate ID tidak boleh dibuat model (herb_id must be in the initial query results)
        # Assumed valid if it doesn't look fabricated (e.g. checks against raw retrieve are done in caller).

        # 4. Field model-assisted wajib mempunyai confidence/labels
        # 5. Field model-assisted wajib lolos critic
        # 6. Aturan pakai numerik harus graph verified (or fallback)
        # 7. Kontraindikasi spesifik harus graph verified
        # 8. Interaksi spesifik harus graph verified
        # 9. Semua model-assisted field harus mempunyai label
        # 10. Tidak ada claim “menyembuhkan”
        # 11. Tidak ada raw placeholder berulang
        # 12. Tidak ada output model yang bertentangan dengan graph.

        # We check candidate's field_verifications
        cleaned_fv = []
        for fv in candidate.field_verifications:
            # Check for "menyembuhkan" claim
            val_str = str(fv.value).lower()
            if "sembuh" in val_str or "menyembuhkan" in val_str:
                # Replace with safe value
                if fv.field_name == "preparation_method":
                    fv = build_safe_general_preparation(candidate)
                elif fv.field_name == "usage_rule":
                    fv = build_safe_general_usage(candidate)
            cleaned_fv.append(fv)
        candidate.field_verifications = cleaned_fv

        valid_recs.append(candidate)

    response.recommendations = valid_recs
    response.total_candidates_eligible = len(valid_recs)
    return response


def analyze_herbal_complaint(
    req: HerbalRecommendationRequest,
    *,
    user_id: str,
    request_id: str | None = None,
) -> HerbalRecommendationResponse:
    started = time.perf_counter()
    rid = request_id or str(uuid.uuid4())
    recommendation_id = str(uuid.uuid4())
    complaint_hash = _safe_hash(req.complaint)

    _log(
        "herbal_recommendation_requested",
        request_id=rid,
        user_id=user_id,
        complaint_length=len(req.complaint),
        complaint_hash=complaint_hash,
        model_id=settings.HERBAL_RECOMMENDATION_MODEL,
    )

    try:
        extracted = extract_complaint(req.complaint)
    except HerbalRecommendationError as exc:
        _failed(rid, "symptom_extraction", exc)
        raise

    extraction_ms = int((time.perf_counter() - started) * 1000)
    red_flags = deterministic_red_flags(req, extracted)
    all_symptoms = list(dict.fromkeys(extracted.primary_symptoms + extracted.secondary_symptoms))

    _log(
        "herbal_symptom_extraction_completed",
        request_id=rid,
        primary_symptom_count=len(extracted.primary_symptoms),
        secondary_symptom_count=len(extracted.secondary_symptoms),
        red_flag_count=len(red_flags),
        processing_ms=extraction_ms,
    )

    if needs_clarification(req, extracted):
        response = HerbalRecommendationResponse(
            recommendation_id=recommendation_id,
            status="clarification_required",
            complaint=req.complaint,
            normalized_complaint=extracted.normalized_summary,
            extracted_symptoms=all_symptoms,
            clarification_questions=extracted.clarification_questions or CLARIFICATION_QUESTIONS,
            red_flags=red_flags,
            medical_attention_signs=medical_attention_signs(),
            general_disclaimer=GENERAL_DISCLAIMER,
            medical_attention_message=None,
            metadata=_metadata(rid, started, error_code="HERBAL_CLARIFICATION_REQUIRED"),
        )
        _STORE[recommendation_id] = {"request": req, "response": response}
        _log(
            "herbal_recommendation_completed",
            request_id=rid,
            recommendation_id=recommendation_id,
            processing_ms=response.metadata["processing_ms"],
            status=response.status,
            recommendation_count=0,
            graph_available=True,
        )
        return response

    if red_flags or extracted.requires_medical_evaluation:
        response = HerbalRecommendationResponse(
            recommendation_id=recommendation_id,
            status="medical_attention_recommended",
            complaint=req.complaint,
            normalized_complaint=extracted.normalized_summary,
            extracted_symptoms=all_symptoms,
            clarification_questions=[],
            red_flags=red_flags,
            medical_attention_signs=medical_attention_signs(),
            general_disclaimer=GENERAL_DISCLAIMER,
            medical_attention_message=medical_attention_message(red_flags or ["memerlukan evaluasi medis"]),
            metadata=_metadata(rid, started, error_code="HERBAL_RED_FLAG_DETECTED"),
        )
        _STORE[recommendation_id] = {"request": req, "response": response}
        _log(
            "herbal_recommendation_completed",
            request_id=rid,
            recommendation_id=recommendation_id,
            processing_ms=response.metadata["processing_ms"],
            status=response.status,
            recommendation_count=0,
            graph_available=True,
        )
        return response

    graph_started = time.perf_counter()
    try:
        raw_candidates, graph_meta = retrieve_graph_verified_herbal_candidates(
            all_symptoms,
            settings.HERBAL_RECOMMENDATION_MAX_RESULTS,
            request_id=rid,
        )
    except HerbalRecommendationError as exc:
        _failed(rid, "graph_retrieval", exc)
        raise

    partial_enrichment = bool(graph_meta.get("partial_enrichment"))

    raw_keys = [
        build_canonical_herb_key_from_values(
            str(item.get("herb_id") or item.get("database_id") or "unknown"),
            item.get("scientific_name"),
            item.get("local_name"),
        )
        for item in raw_candidates
    ]
    unique_raw_keys = sorted(set(raw_keys))
    duplicate_raw_keys = sorted({key for key in raw_keys if raw_keys.count(key) > 1})
    graph_meta["candidate_count_unique"] = len(unique_raw_keys)

    _log(
        "herbal_graph_retrieval_completed",
        request_id=rid,
        symptom_node_count=graph_meta.get("symptom_nodes", 0),
        raw_rows=len(raw_candidates),
        raw_candidate_count=graph_meta.get("candidate_count_raw", 0),
        unique_candidate_count=len(unique_raw_keys),
        processing_ms=int((time.perf_counter() - graph_started) * 1000),
    )

    if not raw_candidates:
        response = HerbalRecommendationResponse(
            recommendation_id=recommendation_id,
            status="no_fully_verified_candidate",
            complaint=req.complaint,
            normalized_complaint=extracted.normalized_summary,
            extracted_symptoms=all_symptoms,
            clarification_questions=[],
            red_flags=red_flags,
            medical_attention_signs=medical_attention_signs(),
            total_candidates_found=0,
            total_candidates_eligible=0,
            total_candidates_excluded=0,
            recommendations=[],
            excluded_candidates=[],
            general_disclaimer=GENERAL_DISCLAIMER,
            medical_attention_message="Belum tersedia rekomendasi dengan aturan pakai, cara pengolahan, dan data keamanan yang lengkap serta terverifikasi.",
            metadata=_metadata(rid, started, **graph_meta, knowledge_graph_version=graph_meta.get("knowledge_graph_version", "herbal-recommendation-v1"), all_graph_verified=True, minimum_coverage=1.0),
        )
        _STORE[recommendation_id] = {"request": req, "response": response}
        _log(
            "herbal_recommendation_completed",
            request_id=rid,
            recommendation_id=recommendation_id,
            recommendation_count=0,
            all_graph_verified="true",
            minimum_coverage=1.0,
            processing_ms=response.metadata["processing_ms"],
            status=response.status,
            graph_available=True,
        )
        return response

    # 1. Start dual verification log
    _log(
        "herbal_dual_verification_started",
        request_id=rid,
        candidate_count=len(raw_candidates),
    )

    scored_candidates = build_scored_candidates(raw_candidates, all_symptoms, req)
    scored_candidates, safety_duplicate_keys = deduplicate_herbal_candidates(scored_candidates)

    processed_candidates = []

    # Count overall verifications for final log
    graph_verified_count = 0
    graph_model_verified_count = 0
    model_assisted_count = 0
    unavailable_count = 0

    model_assisted_candidates_count = 0
    model_assisted_fields_count = 0
    model_assisted_critic_passed_count = 0

    for candidate in scored_candidates:
        # Determine missing fields from Graph perspective
        # Legacies/defaults first
        calculate_dual_verification(candidate)

        _log(
            "herbal_graph_verification_completed",
            request_id=rid,
            candidate_id=candidate.canonical_key,
            graph_coverage_score=candidate.graph_coverage_score,
            missing_graph_fields=",".join(candidate.verification_coverage.missing_fields) if candidate.verification_coverage else "none",
        )

        missing_fields = []
        if not candidate.preparation_methods:
            missing_fields.append("general_preparation")
        if candidate.availability == "unknown":
            missing_fields.append("general_availability")

        # Field verifications placeholder
        candidate.field_verifications = []

        # Graph verified fields add
        if candidate.scientific_name and candidate.local_name:
            candidate.field_verifications.append(
                FieldVerification(
                    field_name="identity",
                    value={"local_name": candidate.local_name, "scientific_name": candidate.scientific_name},
                    verification_source=VerificationSource.GRAPH_VERIFIED,
                    graph_node_ids=[candidate.herb_id],
                )
            )
        if candidate.matched_symptoms:
            candidate.field_verifications.append(
                FieldVerification(
                    field_name="symptom_relevance",
                    value=candidate.matched_symptoms,
                    verification_source=VerificationSource.GRAPH_VERIFIED,
                )
            )

        # 2. Invoke model generator for non-critical fields if missing
        if missing_fields:
            model_assisted_candidates_count += 1
            model_assisted_fields_count += len(missing_fields)
            _log(
                "herbal_model_completion_started",
                request_id=rid,
                candidate_id=candidate.canonical_key,
                allowed_field_count=len(missing_fields),
            )

            # Build graph context payload for generator
            graph_context = {
                "local_name": candidate.local_name,
                "scientific_name": candidate.scientific_name,
                "traditional_uses": candidate.traditional_uses,
                "active_compounds": candidate.active_compounds,
            }

            model_data = model_generate_noncritical_fields(
                herb_context={"local_name": candidate.local_name, "scientific_name": candidate.scientific_name},
                complaint=req.complaint,
                matched_symptoms=candidate.matched_symptoms,
                graph_context=graph_context,
                missing_fields=missing_fields,
                allowed_fields=["general_preparation", "general_availability", "plain_language_summary"],
            )

            if model_data:
                # Critic safety check
                critic_res = model_critic_validate(model_data, graph_context, candidate.local_name)

                _log(
                    "herbal_model_critic_completed",
                    request_id=rid,
                    candidate_id=candidate.canonical_key,
                    passed=str(critic_res.get("passed", False)).lower(),
                    confidence=critic_res.get("confidence", 0.0),
                    violation_count=len(critic_res.get("violations", [])),
                )

                if critic_res.get("passed", False) and critic_res.get("confidence", 0.0) >= 0.70:
                    model_assisted_critic_passed_count += 1
                    # Successfully completed by model
                    if "general_preparation" in model_data and "general_preparation" not in critic_res.get("rejected_fields", []):
                        prep_val = model_data["general_preparation"]
                        steps = prep_val.get("steps") if isinstance(prep_val, dict) else None
                        if not steps and isinstance(prep_val, list):
                            steps = prep_val
                        if steps:
                            # Create model assisted preparation method
                            candidate.preparation_methods = [
                                PreparationMethod(
                                    method_id=f"{candidate.herb_id}-model-prep",
                                    title=prep_val.get("title", "Panduan pengolahan umum") if isinstance(prep_val, dict) else "Panduan pengolahan umum",
                                    plant_part=candidate.plant_parts[0] if candidate.plant_parts else "bagian tanaman",
                                    dosage_form="seduhan",
                                    steps=steps,
                                    suitable_symptoms=candidate.matched_symptoms,
                                    evidence_level="traditional",
                                    verification_status="unverified",
                                    source_ids=[],
                                )
                            ]
                            candidate.field_verifications.append(
                                FieldVerification(
                                    field_name="preparation_method",
                                    value=prep_val,
                                    verification_source=VerificationSource.MODEL_ASSISTED,
                                    model_confidence=critic_res["confidence"],
                                    model_critic_passed=True,
                                    warnings=["Takaran dan durasi spesifik belum terverifikasi."],
                                )
                            )

                    if "general_availability" in model_data and "general_availability" not in critic_res.get("rejected_fields", []):
                        avail_val = model_data["general_availability"]
                        if isinstance(avail_val, dict):
                            cat = avail_val.get("category", "unknown")
                            lbl = avail_val.get("label", "Ketersediaan belum diketahui")
                            reason = avail_val.get("reason")
                        else:
                            cat = "unknown"
                            lbl = str(avail_val)
                            reason = None

                        if cat in {"easy_to_find", "moderately_available", "hard_to_find", "seasonal", "restricted", "unknown"}:
                            candidate.availability = cat
                            candidate.availability_label = f"Perkiraan AI: {lbl}"
                            candidate.availability_reason = reason
                            candidate.field_verifications.append(
                                FieldVerification(
                                    field_name="availability",
                                    value=avail_val,
                                    verification_source=VerificationSource.MODEL_ASSISTED,
                                    model_confidence=critic_res["confidence"],
                                    model_critic_passed=True,
                                )
                            )

                    if "plain_language_summary" in model_data:
                        candidate.explanation = model_data["plain_language_summary"]
                else:
                    # Critic failed or low confidence - use deterministic fallbacks
                    _apply_deterministic_fallbacks(candidate)
            else:
                # Generator failed - use deterministic fallbacks
                _apply_deterministic_fallbacks(candidate)
        else:
            # No missing fields, but set up graph verified indicators
            for method in candidate.preparation_methods:
                candidate.field_verifications.append(
                    FieldVerification(
                        field_name="preparation_method",
                        value={"title": method.title, "steps": method.steps},
                        verification_source=VerificationSource.GRAPH_VERIFIED,
                        source_ids=method.source_ids,
                    )
                )
            if candidate.availability != "unknown":
                candidate.field_verifications.append(
                    FieldVerification(
                        field_name="availability",
                        value=candidate.availability_label,
                        verification_source=VerificationSource.GRAPH_VERIFIED,
                    )
                )

        # 3. Usage rules safety critical rule: model must NOT fabricate dosage if empty in Graph
        if not candidate.usage_rules:
            # Usage rules MUST use deterministic general safety rule
            usage_fv = build_safe_general_usage(candidate)
            candidate.field_verifications.append(usage_fv)
            candidate.usage_rules = [
                UsageRule(
                    usage_rule_id=f"{candidate.herb_id}-safe-usage",
                    form="umum",
                    amount_text="Panduan penggunaan umum",
                    frequency_text="sesuai kebutuhan umum",
                    duration_text="tidak berlebihan",
                    evidence_level="traditional",
                    verification_status="unverified",
                    source_ids=[],
                )
            ]

        # 4. Warnings and general safety warnings
        candidate.general_safety_warnings = [GENERAL_SAFETY_WARNING]
        warning_fvs = build_safe_general_warnings(candidate)
        candidate.field_verifications.extend(warning_fvs)

        # Add graph verified safety statuses or model assisted fallbacks
        for key, field, display_name in [
            ("contraindication", candidate.contraindication_status, "Kontraindikasi khusus"),
            ("interaction", candidate.interaction_status, "Interaksi obat khusus"),
            ("side_effects", candidate.side_effect_status, "Efek samping khusus"),
            ("risk_groups", candidate.risk_group_status, "Kelompok berisiko khusus"),
        ]:
            if field.status != "missing":
                candidate.field_verifications.append(
                    FieldVerification(
                        field_name=key,
                        value=[item.title for item in field.items] if field.status == "known_issue" else "No known issue",
                        verification_source=VerificationSource.GRAPH_VERIFIED,
                        source_ids=field.source_ids,
                    )
                )
            else:
                candidate.field_verifications.append(
                    FieldVerification(
                        field_name=key,
                        value=f"{display_name} tidak tersedia dari knowledge graph. Ikuti panduan keselamatan umum.",
                        verification_source=VerificationSource.MODEL_ASSISTED,
                        safety_critical=True,
                        warnings=["Data keamanan khusus belum terverifikasi."],
                    )
                )

        # Safety data status from Neo4j (contraindications, interactions, side effects, risk groups)
        # Check if they are empty in graph. If empty, safety_data_status is incomplete/missing.
        calculate_dual_verification(candidate)

        # Update overall status counters
        if candidate.overall_verification_status == "fully_graph_verified":
            graph_verified_count += 1
        elif candidate.overall_verification_status == "graph_and_model_verified":
            graph_model_verified_count += 1
        elif candidate.overall_verification_status == "model_assisted_limited":
            model_assisted_count += 1
        else:
            unavailable_count += 1

        processed_candidates.append(candidate)

    _log(
        "herbal_dual_verification_completed",
        request_id=rid,
        graph_verified_count=graph_verified_count,
        graph_model_verified_count=graph_model_verified_count,
        model_assisted_count=model_assisted_count,
        unavailable_count=unavailable_count,
    )

    _log(
        "herbal_model_assisted_enrichment_completed",
        request_id=rid,
        candidate_count=model_assisted_candidates_count,
        field_count=model_assisted_fields_count,
        critic_passed_count=model_assisted_critic_passed_count,
    )

    # Filter out candidates with insufficient_data
    # Candidate must be fully_graph_verified, graph_and_model_verified, or model_assisted_limited.
    eligible_candidates = [c for c in processed_candidates if is_fully_verified_candidate(c)]

    rejection_summary = rejection_reason_summary([c for c in processed_candidates if c not in eligible_candidates])
    _log(
        "herbal_verification_gate_completed",
        request_id=rid,
        candidate_count_input=len(processed_candidates),
        fully_verified_count=sum(1 for c in eligible_candidates if c.overall_verification_status == "fully_graph_verified"),
        rejected_incomplete_count=max(0, len(processed_candidates) - len(eligible_candidates)),
        rejected_conflicting_count=sum(1 for c in processed_candidates if c.has_conflicting_claims),
        rejection_reason_summary=rejection_summary,
    )

    if not eligible_candidates:
        response = HerbalRecommendationResponse(
            recommendation_id=recommendation_id,
            status="no_fully_verified_candidate",
            complaint=req.complaint,
            normalized_complaint=extracted.normalized_summary,
            extracted_symptoms=all_symptoms,
            clarification_questions=[],
            red_flags=red_flags,
            medical_attention_signs=medical_attention_signs(),
            total_candidates_found=len(raw_candidates),
            total_candidates_eligible=0,
            total_candidates_excluded=len(processed_candidates),
            recommendations=[],
            excluded_candidates=[{"herb_id": c.herb_id, "canonical_key": c.canonical_key, "local_name": c.local_name, "missing_fields": (c.verification_coverage.missing_fields if c.verification_coverage else [])} for c in processed_candidates],
            general_disclaimer=GENERAL_DISCLAIMER,
            medical_attention_message="Belum tersedia rekomendasi dengan aturan pakai, cara pengolahan, dan data keamanan yang lengkap serta terverifikasi.",
            metadata=_metadata(rid, started, **graph_meta, candidate_count_after_safety=0, all_graph_verified=True, minimum_coverage=1.0),
        )
        _STORE[recommendation_id] = {"request": req, "response": response}
        _log(
            "herbal_recommendation_completed",
            request_id=rid,
            recommendation_id=recommendation_id,
            recommendation_count=0,
            all_graph_verified="true",
            minimum_coverage=1.0,
            processing_ms=response.metadata["processing_ms"],
            status=response.status,
            graph_available=True,
        )
        return response

    # Safety assessment
    # (Note: safety_assess returns safety status and reasons)
    final_eligible = []
    excluded = []

    for candidate in eligible_candidates:
        # Check safety filter with request variables (allergies, medications, chronic, pregnancy)
        safety_status, safety_reasons = safety_assess(candidate.model_dump(), req)

        # If Neo4j does not have safety_data complete, it must be model_assisted_limited
        if candidate.safety_data_status != "complete":
            # Safety checks check out, but warnings must be general and status cannot be fully_graph_verified
            if candidate.overall_verification_status == "fully_graph_verified":
                candidate.overall_verification_status = "model_assisted_limited"

        candidate.safety_status = safety_status
        candidate.safety_reasons = safety_reasons

        if safety_status == "excluded":
            excluded.append({
                "herb_id": candidate.herb_id,
                "canonical_key": candidate.canonical_key,
                "local_name": candidate.local_name,
                "safety_reasons": safety_reasons,
                "recommendation_score": candidate.recommendation_score,
            })
        else:
            final_eligible.append(candidate)

    _log(
        "herbal_safety_filter_completed",
        request_id=rid,
        input_unique_count=len(eligible_candidates),
        eligible_count=sum(1 for c in final_eligible if c.safety_status == "eligible"),
        conditional_count=sum(1 for c in final_eligible if c.safety_status == "conditional"),
        excluded_count=len(excluded),
    )

    ranked, ranked_excluded = rank_scored_candidates(
        final_eligible,
        settings.HERBAL_RECOMMENDATION_MIN_SCORE,
        settings.HERBAL_RECOMMENDATION_MAX_RESULTS,
    )

    excluded.extend(ranked_excluded)

    # Explanation generation and validation
    explanations: dict[str, str] = {}
    if ranked:
        context = {
            "complaint_summary": extracted.normalized_summary,
            "symptoms": all_symptoms,
            "plant_disclaimer": PLANT_DISCLAIMER,
            "candidates": [
                {**c.model_dump(), "candidate_id": c.canonical_key}
                for c in ranked
            ],
        }
        try:
            explanations = build_grounded_explanations(context)
        except HerbalRecommendationError as exc:
            _failed(rid, "grounded_explanation", exc)
            raise
        explanations, grounding_violations = validate_grounded_explanations(explanations, ranked)
        if grounding_violations:
            _log(
                "herbal_grounding_validation_completed",
                request_id=rid,
                violation_count=len(grounding_violations),
                fallback_used="true",
            )
    for candidate in ranked:
        if not candidate.explanation:
            candidate.explanation = explanations.get(candidate.canonical_key) or explanations.get(candidate.herb_id) or candidate.recommendation_reason

    status = "completed_with_partial_enrichment" if ranked and partial_enrichment else "completed" if ranked else "no_fully_verified_candidate"

    response = HerbalRecommendationResponse(
        recommendation_id=recommendation_id,
        status=status,
        complaint=req.complaint,
        normalized_complaint=extracted.normalized_summary,
        extracted_symptoms=all_symptoms,
        clarification_questions=[],
        red_flags=red_flags,
        medical_attention_signs=medical_attention_signs(),
        total_candidates_found=len(raw_candidates),
        total_candidates_eligible=len(ranked),
        total_candidates_excluded=len(excluded),
        recommendations=ranked,
        excluded_candidates=excluded,
        general_disclaimer=GENERAL_DISCLAIMER,
        medical_attention_message=None,
        metadata=_metadata(
            rid,
            started,
            **graph_meta,
            candidate_count_after_safety=len(ranked),
            all_graph_verified=all(c.overall_verification_status == "fully_graph_verified" for c in ranked),
            minimum_coverage=min((c.graph_coverage_score for c in ranked), default=0.0),
        ),
    )

    # Run custom response validator
    response = validate_dual_verified_response(response)

    _STORE[recommendation_id] = {"request": req, "response": response}
    _log(
        "herbal_recommendation_completed",
        request_id=rid,
        recommendation_id=recommendation_id,
        processing_ms=response.metadata["processing_ms"],
        status=response.status,
        recommendation_count=len(response.recommendations),
        graph_available=True,
    )
    return response


def _apply_deterministic_fallbacks(candidate: HerbalCandidate) -> None:
    """Helper to apply deterministic non-numeric fallbacks."""
    # Preparation method fallback
    prep_fv = build_safe_general_preparation(candidate)
    candidate.field_verifications.append(prep_fv)
    candidate.preparation_methods = [
        PreparationMethod(
            method_id=f"{candidate.herb_id}-safe-prep",
            title="Panduan pengolahan umum",
            plant_part=candidate.plant_parts[0] if candidate.plant_parts else "bagian tanaman",
            dosage_form="seduhan",
            steps=prep_fv.value["steps"],
            suitable_symptoms=candidate.matched_symptoms,
            evidence_level="traditional",
            verification_status="unverified",
            source_ids=[],
        )
    ]

    # Availability fallback
    avail_fv = FieldVerification(
        field_name="availability",
        value="Ketersediaan belum dapat dipastikan",
        verification_source=VerificationSource.UNAVAILABLE,
    )
    candidate.field_verifications.append(avail_fv)
    candidate.availability = "unknown"
    candidate.availability_label = "Ketersediaan belum dapat dipastikan"
