from app.agent.herbal_grounding_validator import validate_grounded_explanations
from app.agent.herbal_ranking import score_candidate
from app.agent.herbal_verification import is_fully_verified_candidate
from app.models.herbal_recommendation import HerbalRecommendationRequest


def safety_section(section_id: str, title: str = "No known issue"):
    return {
        "status": "no_known_issue_within_source_scope",
        "source_ids": ["SRC-A"],
        "items": [],
        "verified_at": "2026-06-10T00:00:00Z",
    }


def verified_raw(**overrides):
    data = {
        "herb_id": "h1",
        "local_name": "Kencur",
        "scientific_name": "Kaempferia galanga L.",
        "aliases": [],
        "matched_symptoms": ["batuk"],
        "traditional_uses": ["batuk"],
        "supported_activities": [],
        "active_compounds": ["ethyl p-methoxycinnamate"],
        "evidence_level": "traditional",
        "availability": "easy_to_find",
        "availability_score": 0.9,
        "availability_reason": "curated verified",
        "source_ids": ["SRC-A"],
        "sources": [{"id": "SRC-A", "title": "Verified monograph", "publisher": "Publisher", "year": 2024, "qualityGrade": "A", "active": True}],
        "graph_node_ids": ["h1", "use1", "prep1", "usage1", "availability1", "contra1", "interaction1", "side1", "risk1", "warning1"],
        "toxicity": [],
        "contraindications_status": safety_section("contra"),
        "interactions_status": safety_section("interaction"),
        "side_effects_status": safety_section("side"),
        "risk_groups_status": safety_section("risk"),
        "contraindications": [],
        "interactions": [],
        "side_effects": [],
        "risk_groups": [],
        "warnings": [{"id": "WARN-1", "title": "Hentikan jika muncul reaksi alergi", "description": "Hentikan penggunaan bila muncul reaksi alergi yang tercatat.", "severity": "moderate", "action_text": "Hentikan penggunaan dan cari bantuan bila berat.", "source_ids": ["SRC-A"]}],
        "stop_use_signs": ["muncul reaksi alergi"],
        "preparation_methods": [{
            "method_id": "prep1",
            "title": "Seduhan rimpang",
            "plant_part": "rimpang",
            "dosage_form": "seduhan",
            "ingredients": [{"name": "rimpang kencur", "amount_text": "sesuai monografi", "source_ids": ["SRC-A"]}],
            "steps": ["Cuci rimpang hingga bersih.", "Potong rimpang sesuai petunjuk monografi.", "Seduh sesuai durasi pada sumber.", "Saring sebelum digunakan."],
            "suitable_symptoms": ["batuk"],
            "evidence_level": "traditional",
            "verification_status": "verified",
            "source_ids": ["SRC-A"],
            "source": "SRC-A",
        }],
        "usage_rules": [{
            "usage_rule_id": "usage1",
            "form": "seduhan",
            "amount_text": "100 ml",
            "frequency_text": "1 kali sehari",
            "duration_text": "1 hari",
            "allowed_age_groups": ["adult"],
            "prohibited_age_groups": ["infant"],
            "evidence_level": "traditional",
            "verification_status": "verified",
            "source_ids": ["SRC-A"],
            "source": "SRC-A",
        }],
    }
    data.update(overrides)
    return data


def test_full_coverage_candidate_passes_gate():
    candidate = score_candidate(verified_raw(), ["batuk"], HerbalRecommendationRequest(complaint="batuk"))
    assert is_fully_verified_candidate(candidate)
    assert candidate.verification_coverage.coverage_score == 1.0
    assert candidate.provenance.graph_verified is True


def test_missing_preparation_fails_gate():
    candidate = score_candidate(verified_raw(preparation_methods=[]), ["batuk"], HerbalRecommendationRequest(complaint="batuk"))
    assert not is_fully_verified_candidate(candidate)
    assert "preparation" in candidate.verification_coverage.missing_fields


def test_missing_usage_rule_fails_gate():
    candidate = score_candidate(verified_raw(usage_rules=[]), ["batuk"], HerbalRecommendationRequest(complaint="batuk"))
    assert not is_fully_verified_candidate(candidate)
    assert "usage_rule" in candidate.verification_coverage.missing_fields


def test_missing_interaction_evidence_fails_gate():
    candidate = score_candidate(verified_raw(interactions_status={"status": "missing", "items": [], "source_ids": []}), ["batuk"], HerbalRecommendationRequest(complaint="batuk"))
    assert not is_fully_verified_candidate(candidate)
    assert "interaction_status" in candidate.verification_coverage.missing_fields


def test_empty_list_without_source_is_missing():
    candidate = score_candidate(verified_raw(side_effects_status={"status": "no_known_issue_within_source_scope", "items": [], "source_ids": []}), ["batuk"], HerbalRecommendationRequest(complaint="batuk"))
    assert not is_fully_verified_candidate(candidate)
    assert "side_effect_status" in candidate.verification_coverage.missing_fields


def test_conflicting_safety_status_fails_gate():
    candidate = score_candidate(verified_raw(risk_groups_status={"status": "conflicting", "items": [], "source_ids": ["SRC-A"]}), ["batuk"], HerbalRecommendationRequest(complaint="batuk"))
    assert not is_fully_verified_candidate(candidate)


def test_llm_hallucinated_ingredient_rejected():
    candidate = score_candidate(verified_raw(), ["batuk"], HerbalRecommendationRequest(complaint="batuk"))
    assert is_fully_verified_candidate(candidate)
    explanations, violations = validate_grounded_explanations({candidate.canonical_key: "Tambahkan madu 2 sendok lalu minum."}, [candidate])
    assert violations
    assert "madu" not in explanations[candidate.canonical_key].casefold()
