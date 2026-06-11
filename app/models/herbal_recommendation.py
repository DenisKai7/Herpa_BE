"""Schemas for grounded herbal recommendation flow with dual verification."""

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


SafetyStatus = Literal[
    "known_issue",
    "no_known_issue_within_source_scope",
    "missing",
    "conflicting",
]


# ---------------------------------------------------------------------------
# Dual-verification enums and metadata
# ---------------------------------------------------------------------------

class VerificationSource(str, Enum):
    GRAPH_VERIFIED = "graph_verified"
    GRAPH_MODEL_VERIFIED = "graph_model_verified"
    MODEL_ASSISTED = "model_assisted"
    UNAVAILABLE = "unavailable"


class FieldVerification(BaseModel):
    field_name: str
    value: Any
    verification_source: VerificationSource
    graph_node_ids: list[str] = Field(default_factory=list)
    graph_relationship_ids: list[str] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)
    model_confidence: float | None = None
    model_critic_passed: bool = False
    safety_critical: bool = False
    warnings: list[str] = Field(default_factory=list)


# Safety-critical fields: model must NOT fabricate data for these.
SAFETY_CRITICAL_FIELDS = frozenset({
    "dosage_numeric",
    "frequency",
    "duration",
    "max_consumption",
    "contraindications",
    "interactions",
    "pregnancy_use",
    "breastfeeding_use",
    "infant_use",
    "child_use",
    "elderly_use",
    "liver_disease",
    "kidney_disease",
    "bleeding_disorder",
    "side_effects_specific",
    "toxicity",
    "prescription_drug_combination",
})

# Non-critical fields: model MAY provide fallback.
MODEL_ALLOWED_FIELDS = frozenset({
    "general_preparation",
    "general_availability",
    "plain_language_summary",
    "recommendation_reason_summary",
    "hygiene_steps",
    "traditional_use_description",
    "term_explanation",
})


# ---------------------------------------------------------------------------
# Graph coverage scoring weights
# ---------------------------------------------------------------------------

GRAPH_COVERAGE_WEIGHTS: dict[str, float] = {
    "identity": 0.15,
    "symptom_relevance": 0.15,
    "therapeutic_use": 0.10,
    "active_compounds": 0.10,
    "preparation": 0.10,
    "usage_rule": 0.10,
    "contraindication": 0.08,
    "interaction": 0.08,
    "side_effects": 0.05,
    "risk_groups": 0.04,
    "availability": 0.03,
    "provenance": 0.02,
}


# ---------------------------------------------------------------------------
# General safety warnings (deterministic, always shown)
# ---------------------------------------------------------------------------

GENERAL_SAFETY_WARNING = (
    "Konsultasikan dengan tenaga kesehatan sebelum menggunakan herbal jika Anda:\n"
    "- sedang hamil atau menyusui;\n"
    "- memberikan herbal kepada bayi atau anak;\n"
    "- berusia lanjut dengan beberapa penyakit;\n"
    "- mempunyai penyakit hati atau ginjal;\n"
    "- mempunyai gangguan perdarahan;\n"
    "- mempunyai alergi tanaman;\n"
    "- menggunakan obat rutin;\n"
    "- akan menjalani operasi."
)


# ---------------------------------------------------------------------------
# Original schemas (preserved for compatibility)
# ---------------------------------------------------------------------------

class HerbalRecommendationRequest(BaseModel):
    complaint: str = Field(min_length=3, max_length=1000)
    age_group: Literal[
        "unknown", "infant", "child", "adolescent", "adult", "elderly"
    ] = "unknown"
    pregnancy_status: Literal[
        "unknown", "not_pregnant", "pregnant", "breastfeeding"
    ] = "unknown"
    allergies: list[str] = Field(default_factory=list)
    chronic_conditions: list[str] = Field(default_factory=list)
    current_medications: list[str] = Field(default_factory=list)


class ExtractedComplaint(BaseModel):
    original_text: str
    normalized_summary: str
    primary_symptoms: list[str] = Field(default_factory=list)
    secondary_symptoms: list[str] = Field(default_factory=list)
    body_systems: list[str] = Field(default_factory=list)
    duration_text: str | None = None
    severity: Literal["unknown", "mild", "moderate", "severe"] = "unknown"
    red_flags: list[str] = Field(default_factory=list)
    possible_intents: list[str] = Field(default_factory=list)
    requires_medical_evaluation: bool = False
    clarification_questions: list[str] = Field(default_factory=list)


class IngredientItem(BaseModel):
    name: str
    amount_text: str | None = None
    source_ids: list[str] = Field(default_factory=list)


class PreparationMethod(BaseModel):
    method_id: str
    title: str
    plant_part: str
    dosage_form: str
    ingredients: list[IngredientItem] = Field(default_factory=list)
    steps: list[str] = Field(default_factory=list)
    water_volume_text: str | None = None
    temperature_text: str | None = None
    preparation_duration_text: str | None = None
    storage_instruction: str | None = None
    suitable_symptoms: list[str] = Field(default_factory=list)
    evidence_level: str
    verification_status: str
    source_ids: list[str] = Field(default_factory=list)
    # Legacy compatibility aliases populated only from graph data.
    preparation_type: str = "data_not_available"
    source: str | None = None
    compatible_symptoms: list[str] = Field(default_factory=list)
    contraindicated_groups: list[str] = Field(default_factory=list)


class UsageRule(BaseModel):
    usage_rule_id: str
    form: str | None = None
    amount_text: str
    frequency_text: str
    administration_time_text: str | None = None
    duration_text: str
    maximum_duration_text: str | None = None
    before_or_after_meal: str | None = None
    administration_notes: list[str] = Field(default_factory=list)
    allowed_age_groups: list[str] = Field(default_factory=list)
    prohibited_age_groups: list[str] = Field(default_factory=list)
    applicable_age_groups: list[str] = Field(default_factory=list)
    evidence_level: str
    verification_status: str
    source_ids: list[str] = Field(default_factory=list)
    source: str | None = None


class SafetyItem(BaseModel):
    safety_id: str
    title: str
    description: str
    severity: str
    action_text: str
    source_ids: list[str] = Field(default_factory=list)
    # Legacy display aliases.
    id: str | None = None
    label: str = ""


class VerifiedSafetyField(BaseModel):
    status: SafetyStatus = "missing"
    items: list[SafetyItem] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)
    verified_at: str | None = None


class AvailabilityInfo(BaseModel):
    category: Literal["easy_to_find", "moderately_available", "hard_to_find", "seasonal", "restricted"]
    label: str
    reason: str
    source_ids: list[str]


class EvidenceInfo(BaseModel):
    level: str
    label: str
    source_ids: list[str]


class SourceProvenanceItem(BaseModel):
    source_id: str
    title: str
    publisher: str | None = None
    year: int | None = None
    evidence_grade: str | None = None
    url: str | None = None
    verified_at: str | None = None
    active: bool = True


class HerbVerificationCoverage(BaseModel):
    herb_id: str
    identity_verified: bool = False
    therapeutic_use_verified: bool = False
    preparation_verified: bool = False
    usage_rule_verified: bool = False
    contraindication_verified: bool = False
    interaction_verified: bool = False
    side_effect_verified: bool = False
    risk_group_verified: bool = False
    warning_verified: bool = False
    availability_verified: bool = False
    provenance_verified: bool = False
    verified_field_count: int = 0
    required_field_count: int = 11
    coverage_score: float = 0.0
    source_ids: list[str] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)


class GraphProvenance(BaseModel):
    graph_verified: bool = False
    coverage_score: float = 0.0
    source_ids: list[str] = Field(default_factory=list)
    sources: list[SourceProvenanceItem] = Field(default_factory=list)
    evidence_claim_ids: list[str] = Field(default_factory=list)
    graph_node_ids: list[str] = Field(default_factory=list)
    graph_relationship_ids: list[str] = Field(default_factory=list)
    verified_at: str | None = None
    data_version: str = "herbal-recommendation-v1"


class HerbalCandidate(BaseModel):
    herb_id: str
    canonical_key: str
    source_herb_ids: list[str] = Field(default_factory=list)
    local_name: str
    scientific_name: str | None = None
    aliases: list[str] = Field(default_factory=list)
    matched_symptoms: list[str] = Field(default_factory=list)
    unmatched_symptoms: list[str] = Field(default_factory=list)
    recommendation_reason: str = ""
    plant_parts: list[str] = Field(default_factory=list)
    active_compounds: list[str] = Field(default_factory=list)
    traditional_uses: list[str] = Field(default_factory=list)
    supported_activities: list[str] = Field(default_factory=list)
    evidence_level: str = "data_not_available"
    preparation_methods: list[PreparationMethod] = Field(default_factory=list)
    usage_rules: list[UsageRule] = Field(default_factory=list)
    contraindications: list[str] = Field(default_factory=list)
    interactions: list[str] = Field(default_factory=list)
    side_effects: list[str] = Field(default_factory=list)
    risk_groups: list[str] = Field(default_factory=list)
    warnings: list[SafetyItem] = Field(default_factory=list)
    stop_use_signs: list[str] = Field(default_factory=list)
    medical_attention_signs: list[str] = Field(default_factory=list)
    availability: Literal[
        "easy_to_find", "moderately_available", "hard_to_find", "seasonal", "restricted", "unknown"
    ] = "unknown"
    availability_label: str = "Ketersediaan belum diketahui"
    availability_reason: str | None = None
    recommendation_score: float = 0.0
    safety_status: Literal["eligible", "conditional", "excluded"] = "conditional"
    safety_reasons: list[str] = Field(default_factory=list)
    explanation: str | None = None
    usage_status: Literal["available", "insufficient_data"] = "insufficient_data"
    graph_verified: bool = False
    provenance_valid: bool = False
    has_conflicting_claims: bool = False
    verification_coverage: HerbVerificationCoverage | None = None
    provenance: GraphProvenance | None = None
    availability_info: AvailabilityInfo | None = None
    evidence: EvidenceInfo | None = None
    contraindication_status: VerifiedSafetyField = Field(default_factory=VerifiedSafetyField)
    interaction_status: VerifiedSafetyField = Field(default_factory=VerifiedSafetyField)
    side_effect_status: VerifiedSafetyField = Field(default_factory=VerifiedSafetyField)
    risk_group_status: VerifiedSafetyField = Field(default_factory=VerifiedSafetyField)

    # --- Dual-verification fields ---
    field_verifications: list[FieldVerification] = Field(default_factory=list)
    graph_coverage_score: float = 0.0
    model_assisted_coverage_score: float = 0.0
    overall_verification_status: Literal[
        "fully_graph_verified",
        "graph_and_model_verified",
        "model_assisted_limited",
        "insufficient_data",
    ] = "insufficient_data"
    safety_data_status: Literal["complete", "incomplete", "missing"] = "missing"
    general_safety_warnings: list[str] = Field(default_factory=list)


class HerbalRecommendationResponse(BaseModel):
    recommendation_id: str
    status: Literal[
        "completed",
        "completed_with_partial_enrichment",
        "clarification_required",
        "medical_attention_recommended",
        "no_safe_candidate",
        "no_fully_verified_candidate",
        "graph_unavailable",
        "failed",
    ]
    complaint: str
    normalized_complaint: str
    extracted_symptoms: list[str] = Field(default_factory=list)
    clarification_questions: list[str] = Field(default_factory=list)
    red_flags: list[str] = Field(default_factory=list)
    medical_attention_signs: list[str] = Field(default_factory=list)
    total_candidates_found: int = 0
    total_candidates_eligible: int = 0
    total_candidates_excluded: int = 0
    recommendations: list[HerbalCandidate] = Field(default_factory=list)
    excluded_candidates: list[dict[str, Any]] = Field(default_factory=list)
    general_disclaimer: str
    medical_attention_message: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class HerbalRecommendationError(Exception):
    def __init__(self, code: str, message: str, status_code: int = 500, retryable: bool = False):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.retryable = retryable
