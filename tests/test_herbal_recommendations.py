import logging
import pytest
from fastapi.testclient import TestClient

from app.core.dependencies import verify_user
from app.main import app
from app.models.herbal_recommendation import ExtractedComplaint, HerbalRecommendationError, VerificationSource
from app.core.database import neo4j_driver


def mock_verify_user() -> str:
    return "test-user-id"


@pytest.fixture
def client():
    app.dependency_overrides[verify_user] = mock_verify_user
    yield TestClient(app)
    app.dependency_overrides.clear()


def extracted(**overrides):
    data = {
        "original_text": "batuk ringan",
        "normalized_summary": "batuk ringan",
        "primary_symptoms": ["batuk"],
        "secondary_symptoms": [],
        "body_systems": ["pernapasan"],
        "duration_text": None,
        "severity": "mild",
        "red_flags": [],
        "possible_intents": ["herbal_support"],
        "requires_medical_evaluation": False,
        "clarification_questions": [],
    }
    data.update(overrides)
    return ExtractedComplaint(**data)


def safety_status():
    return {"status": "no_known_issue_within_source_scope", "items": [], "source_ids": ["SRC-TEST"], "verified_at": "2026-06-10T00:00:00Z"}


def raw_candidate(idx: int, **overrides):
    data = {
        "herb_id": f"h{idx}",
        "local_name": f"Herbal {idx}",
        "scientific_name": f"Plantus {idx}",
        "aliases": [],
        "matched_symptoms": ["batuk"],
        "traditional_uses": ["batuk"],
        "supported_activities": ["tradisional"],
        "active_compounds": ["senyawa terverifikasi"],
        "evidence_level": "traditional",
        "availability": "easy_to_find",
        "availability_score": 0.8,
        "availability_reason": "curated",
        "source_ids": ["SRC-TEST"],
        "sources": [{"id": "SRC-TEST", "title": "Verified test source", "publisher": "Publisher", "year": 2024, "qualityGrade": "A", "active": True}],
        "graph_node_ids": [f"h{idx}", f"use{idx}", f"prep{idx}", f"usage{idx}", f"avail{idx}", f"contra{idx}", f"interaction{idx}", f"side{idx}", f"risk{idx}", f"warning{idx}"],
        "toxicity": [],
        "contraindications_status": safety_status(),
        "interactions_status": safety_status(),
        "side_effects_status": safety_status(),
        "risk_groups_status": safety_status(),
        "contraindications": [],
        "interactions": [],
        "side_effects": [],
        "risk_groups": [],
        "warnings": [{"id": f"warn{idx}", "title": "Hentikan jika muncul reaksi alergi", "description": "Hentikan penggunaan bila muncul reaksi alergi.", "severity": "moderate", "action_text": "Hentikan penggunaan.", "source_ids": ["SRC-TEST"]}],
        "stop_use_signs": ["muncul reaksi alergi"],
        "preparation_methods": [
            {
                "method_id": f"p{idx}",
                "title": "Seduhan data graph",
                "plant_part": "rimpang",
                "dosage_form": "seduhan",
                "steps": ["Cuci bahan dari graph.", "Potong bahan dari graph.", "Seduh sesuai sumber graph.", "Saring hasil seduhan."],
                "ingredients": [{"name": "Bahan dari graph", "amount_text": "sesuai sumber", "source_ids": ["SRC-TEST"]}],
                "suitable_symptoms": ["batuk"],
                "evidence_level": "traditional",
                "verification_status": "verified",
                "source_ids": ["SRC-TEST"],
                "source": "SRC-TEST",
            }
        ],
        "usage_rules": [
            {
                "usage_rule_id": f"u{idx}",
                "form": "seduhan",
                "amount_text": "data graph",
                "frequency_text": "data graph",
                "duration_text": "data graph",
                "allowed_age_groups": ["adult"],
                "prohibited_age_groups": ["infant"],
                "evidence_level": "traditional",
                "verification_status": "verified",
                "source_ids": ["SRC-TEST"],
                "source": "SRC-TEST",
            }
        ],
    }
    data.update(overrides)
    return data


def patch_success(monkeypatch, count=6, **candidate_overrides):
    from app.agent import herbal_recommendation_service as service

    monkeypatch.setattr(service, "extract_complaint", lambda complaint: extracted(original_text=complaint))
    monkeypatch.setattr(
        service,
        "retrieve_graph_verified_herbal_candidates",
        lambda symptoms, max_results, request_id=None: ([raw_candidate(i, **candidate_overrides) for i in range(count)], {
            "symptom_nodes": 1,
            "graph_records": count,
            "candidate_count_raw": count,
            "knowledge_graph_version": "herbal-recommendation-v1",
        }),
    )
    monkeypatch.setattr(
        service,
        "build_grounded_explanations",
        lambda context: {c["candidate_id"]: "Membantu meredakan secara tradisional berdasarkan knowledge graph." for c in context["candidates"]},
    )
    monkeypatch.setattr(
        service,
        "model_generate_noncritical_fields",
        lambda *a, **k: {
            "general_preparation": {"title": "Seduhan model", "steps": ["Cuci bersih bahan", "Seduh air panas"]},
            "general_availability": {"category": "easy_to_find", "label": "Mudah dicari", "reason": "Banyak di pasar"},
            "plain_language_summary": "Grounded model summary.",
        },
    )
    monkeypatch.setattr(
        service,
        "model_critic_validate",
        lambda *a, **k: {
            "passed": True,
            "violations": [],
            "safe_fields": ["general_preparation", "general_availability"],
            "rejected_fields": [],
            "confidence": 0.95,
        },
    )


def test_route_registered():
    routes = {(route.path, tuple(sorted(route.methods or []))) for route in app.routes}
    assert any(path == "/api/herbal-recommendations/analyze" and "POST" in methods for path, methods in routes)
    assert any(path == "/api/herbal-recommendations/health" and "GET" in methods for path, methods in routes)


def test_herbal_health(client):
    response = client.get("/api/herbal-recommendations/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "feature": "herbal_recommendations",
        "analyze_endpoint": "/api/herbal-recommendations/analyze",
    }


def test_analyze_returns_all_eligible_and_logs(client, monkeypatch, caplog):
    patch_success(monkeypatch, count=6)
    caplog.set_level(logging.INFO)

    response = client.post(
        "/api/herbal-recommendations/analyze",
        json={"complaint": "batuk ringan", "age_group": "adult"},
        headers={"X-Request-ID": "rid-test"},
    )

    assert response.status_code == 200
    body = response.json()
    assert response.headers["x-request-id"] == "rid-test"
    assert body["status"] == "completed"
    assert len(body["recommendations"]) == 6
    logs = caplog.text
    for event in [
        "herbal_recommendation_requested",
        "herbal_symptom_extraction_completed",
        "herbal_graph_retrieval_completed",
        "herbal_safety_filter_completed",
        "herbal_recommendation_completed",
        "herbal_dual_verification_started",
        "herbal_graph_verification_completed",
        "herbal_dual_verification_completed",
    ]:
        assert event in logs


# Wajib 1. Graph lengkap (Expectation: fully_graph_verified)
def test_wajib_graph_lengkap(client, monkeypatch):
    patch_success(monkeypatch, count=1)
    response = client.post("/api/herbal-recommendations/analyze", json={"complaint": "batuk ringan"})
    assert response.status_code == 200
    candidate = response.json()["recommendations"][0]
    assert candidate["overall_verification_status"] == "fully_graph_verified"


# Wajib 2. Graph preparation kosong (Expectation: model_assisted, no fabricated dose)
def test_wajib_graph_preparation_kosong(client, monkeypatch):
    # Pass preparation_methods=[] to simulate missing graph preparation data
    patch_success(monkeypatch, count=1, preparation_methods=[])
    response = client.post("/api/recommendations" if False else "/api/herbal-recommendations/analyze", json={"complaint": "batuk ringan"})
    assert response.status_code == 200
    candidate = response.json()["recommendations"][0]
    assert candidate["overall_verification_status"] == "model_assisted_limited"
    # Check that it contains "model_assisted" for preparation_method
    prep_fv = next(fv for fv in candidate["field_verifications"] if fv["field_name"] == "preparation_method")
    assert prep_fv["verification_source"] == "model_assisted"
    assert "dosis" not in str(prep_fv["value"]).lower()


# Wajib 3. Graph usage rule kosong (Expectation: specific_dosage_available=false)
def test_wajib_graph_usage_rule_kosong(client, monkeypatch):
    patch_success(monkeypatch, count=1, usage_rules=[])
    response = client.post("/api/herbal-recommendations/analyze", json={"complaint": "batuk ringan"})
    assert response.status_code == 200
    candidate = response.json()["recommendations"][0]
    usage_fv = next(fv for fv in candidate["field_verifications"] if fv["field_name"] == "usage_rule")
    assert usage_fv["value"]["specific_dosage_available"] is False
    assert "Aturan pakai spesifik belum mempunyai sumber terverifikasi." in usage_fv["warnings"][0]


# Wajib 4. Graph safety kosong (Expectation: show general warnings, no fabricated contraindications)
def test_wajib_graph_safety_kosong(client, monkeypatch):
    patch_success(
        monkeypatch,
        count=1,
        contraindications_status={"status": "missing", "items": [], "source_ids": []},
        interactions_status={"status": "missing", "items": [], "source_ids": []},
        side_effects_status={"status": "missing", "items": [], "source_ids": []},
        risk_groups_status={"status": "missing", "items": [], "source_ids": []},
    )
    response = client.post("/api/herbal-recommendations/analyze", json={"complaint": "batuk ringan"})
    assert response.status_code == 200
    candidate = response.json()["recommendations"][0]
    # General warning must be present
    assert len(candidate["general_safety_warnings"]) > 0
    assert "Konsultasikan dengan tenaga kesehatan" in candidate["general_safety_warnings"][0]


# Wajib 5. Model membuat dosis (Critic must reject)
def test_wajib_model_membuat_dosis_rejected(client, monkeypatch):
    patch_success(monkeypatch, count=1, preparation_methods=[])
    from app.agent import herbal_recommendation_service as service
    # Make generator return dosage and critic reject it
    monkeypatch.setattr(
        service,
        "model_generate_noncritical_fields",
        lambda *a, **k: {"general_preparation": {"title": "Seduhan", "steps": ["Minum 3 kali sehari 100ml"]}},
    )
    monkeypatch.setattr(
        service,
        "model_critic_validate",
        lambda *a, **k: {
            "passed": False,
            "violations": ["fabricated_dosage"],
            "safe_fields": [],
            "rejected_fields": ["general_preparation"],
            "confidence": 0.90,
        },
    )
    response = client.post("/api/herbal-recommendations/analyze", json={"complaint": "batuk ringan"})
    assert response.status_code == 200
    candidate = response.json()["recommendations"][0]
    # Fallback must be used
    prep_fv = next(fv for fv in candidate["field_verifications"] if fv["field_name"] == "preparation_method")
    assert prep_fv["verification_source"] == "model_assisted"
    assert prep_fv["model_critic_passed"] is False  # Fallback did not pass critic (passed=False)


# Wajib 6. Model membuat interaksi obat (Critic must reject)
def test_wajib_model_membuat_interaksi_rejected(client, monkeypatch):
    patch_success(monkeypatch, count=1, preparation_methods=[])
    from app.agent import herbal_recommendation_service as service
    monkeypatch.setattr(
        service,
        "model_generate_noncritical_fields",
        lambda *a, **k: {"general_preparation": {"title": "Seduhan", "steps": ["Berinteraksi dengan aspirin"]}},
    )
    monkeypatch.setattr(
        service,
        "model_critic_validate",
        lambda *a, **k: {
            "passed": False,
            "violations": ["fabricated_interaction"],
            "safe_fields": [],
            "rejected_fields": ["general_preparation"],
            "confidence": 0.90,
        },
    )
    response = client.post("/api/herbal-recommendations/analyze", json={"complaint": "batuk ringan"})
    assert response.status_code == 200
    candidate = response.json()["recommendations"][0]
    prep_fv = next(fv for fv in candidate["field_verifications"] if fv["field_name"] == "preparation_method")
    assert prep_fv["model_critic_passed"] is False


# Wajib 7. Model menyebut tanaman lain (Critic must reject)
def test_wajib_model_tanaman_lain_rejected(client, monkeypatch):
    patch_success(monkeypatch, count=1, preparation_methods=[])
    from app.agent import herbal_recommendation_service as service
    monkeypatch.setattr(
        service,
        "model_generate_noncritical_fields",
        lambda *a, **k: {"general_preparation": {"title": "Gunakan Jahe", "steps": ["Jahe bermanfaat..."]}},
    )
    monkeypatch.setattr(
        service,
        "model_critic_validate",
        lambda *a, **k: {
            "passed": False,
            "violations": ["different_plant_name"],
            "safe_fields": [],
            "rejected_fields": ["general_preparation"],
            "confidence": 0.90,
        },
    )
    response = client.post("/api/herbal-recommendations/analyze", json={"complaint": "batuk ringan"})
    assert response.status_code == 200
    candidate = response.json()["recommendations"][0]
    prep_fv = next(fv for fv in candidate["field_verifications"] if fv["field_name"] == "preparation_method")
    assert prep_fv["model_critic_passed"] is False


# Wajib 8. Model membuat klaim menyembuhkan (Critic must reject)
def test_wajib_model_klaim_menyembuhkan_rejected(client, monkeypatch):
    patch_success(monkeypatch, count=1, preparation_methods=[])
    from app.agent import herbal_recommendation_service as service
    # Test validator replacing claim as well
    monkeypatch.setattr(
        service,
        "model_generate_noncritical_fields",
        lambda *a, **k: {"general_preparation": {"title": "Seduhan", "steps": ["Pasti menyembuhkan batuk 100%"]}},
    )
    monkeypatch.setattr(
        service,
        "model_critic_validate",
        lambda *a, **k: {
            "passed": True,  # Supposing critic missed it, validator must clean it
            "violations": [],
            "safe_fields": ["general_preparation"],
            "rejected_fields": [],
            "confidence": 0.95,
        },
    )
    response = client.post("/api/herbal-recommendations/analyze", json={"complaint": "batuk ringan"})
    assert response.status_code == 200
    candidate = response.json()["recommendations"][0]
    prep_fv = next(fv for fv in candidate["field_verifications"] if fv["field_name"] == "preparation_method")
    assert "menyembuhkan" not in str(prep_fv["value"]).lower()


# Wajib 9. Model critic gagal (Use deterministic fallback, request 200 OK)
def test_wajib_model_critic_gagal(client, monkeypatch):
    patch_success(monkeypatch, count=1, preparation_methods=[])
    from app.agent import herbal_recommendation_service as service
    monkeypatch.setattr(
        service,
        "model_critic_validate",
        lambda *a, **k: {"passed": False, "violations": ["critic_error"], "safe_fields": [], "rejected_fields": [], "confidence": 0.0},
    )
    response = client.post("/api/herbal-recommendations/analyze", json={"complaint": "batuk ringan"})
    assert response.status_code == 200
    candidate = response.json()["recommendations"][0]
    prep_fv = next(fv for fv in candidate["field_verifications"] if fv["field_name"] == "preparation_method")
    assert prep_fv["verification_source"] == VerificationSource.MODEL_ASSISTED
    assert prep_fv["model_critic_passed"] is False


# Wajib 10. Neo4j gagal (Do not recommend, raise error or return failed status)
def test_wajib_neo4j_gagal(client, monkeypatch):
    from app.agent import herbal_recommendation_service as service
    monkeypatch.setattr(service, "extract_complaint", lambda complaint: extracted(original_text=complaint))
    def fail_graph(symptoms, max_results, request_id=None):
        raise HerbalRecommendationError("HERBAL_GRAPH_UNAVAILABLE", "Neo4j connection error", status_code=503)
    monkeypatch.setattr(service, "retrieve_graph_verified_herbal_candidates", fail_graph)

    response = client.post("/api/herbal-recommendations/analyze", json={"complaint": "batuk ringan"})
    assert response.status_code == 503
    assert response.json()["detail"]["error"]["code"] == "HERBAL_GRAPH_UNAVAILABLE"


# Wajib 12. Exact complaint: "batuk berdahak, tenggorokan gatal"
def test_wajib_exact_complaint(client, monkeypatch):
    patch_success(monkeypatch, count=1, local_name="Kencur")
    from app.agent import herbal_recommendation_service as service
    monkeypatch.setattr(service, "extract_complaint", lambda complaint: extracted(
        original_text=complaint,
        normalized_summary="batuk berdahak, tenggorokan gatal",
        primary_symptoms=["batuk", "gatal"],
    ))

    response = client.post("/api/herbal-recommendations/analyze", json={"complaint": "batuk berdahak, tenggorokan gatal"})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    candidate = body["recommendations"][0]
    assert candidate["local_name"] == "Kencur"
    assert "belum lolos provenance" not in str(body).lower()
    # Peringatan umum
    assert any("tenaga kesehatan" in warning for warning in candidate["general_safety_warnings"])


def test_healthcheck_herbal_graph_success(client, monkeypatch):
    monkeypatch.setattr(
        neo4j_driver,
        "execute_query",
        lambda *args, **kwargs: ([type("Record", (), {"data": lambda self: {"ok": 1}})()], None, None),
    )
    from app.api import herbal_recommendations as api_mod
    from app.agent.herbal_graph_schema import HerbalGraphSchema
    monkeypatch.setattr(
        api_mod,
        "load_herbal_graph_schema",
        lambda: HerbalGraphSchema(available_relationships=["USED_FOR", "HAS_TOXICITY"]),
    )

    response = client.get("/api/health/herbal-graph")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["connected"] is True
    assert body["base_retrieval_ready"] is True
    assert body["capabilities"]["therapeutic_use"] is True
    assert body["capabilities"]["preparation"] is False


def test_healthcheck_herbal_graph_failure(client, monkeypatch):
    def fail_query(*args, **kwargs):
        raise Exception("Neo4j down")
    monkeypatch.setattr(neo4j_driver, "execute_query", fail_query)

    response = client.get("/api/health/herbal-graph")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    assert body["connected"] is False
    assert body["base_retrieval_ready"] is False


def test_classify_neo4j_error():
    from app.agent.herbal_graph_schema import classify_neo4j_error
    from neo4j.exceptions import AuthError, ClientError, ServiceUnavailable, TransientError

    assert classify_neo4j_error(AuthError("invalid auth")) == "HERBAL_GRAPH_AUTH_FAILED"
    assert classify_neo4j_error(ServiceUnavailable("host unreachable")) == "HERBAL_GRAPH_CONNECTION_FAILED"
    assert classify_neo4j_error(TransientError("database timeout")) == "HERBAL_GRAPH_TIMEOUT"
    assert classify_neo4j_error(ClientError("syntax error")) == "HERBAL_GRAPH_QUERY_INVALID"
    assert classify_neo4j_error(Exception("query timed out after 5s")) == "HERBAL_GRAPH_TIMEOUT"


def test_partial_enrichment_response_status(client, monkeypatch):
    patch_success(monkeypatch, count=1)
    from app.agent import herbal_recommendation_service as service
    monkeypatch.setattr(
        service,
        "retrieve_graph_verified_herbal_candidates",
        lambda symptoms, max_results, request_id=None: ([raw_candidate(0)], {
            "symptom_nodes": 1,
            "graph_records": 1,
            "candidate_count_raw": 1,
            "knowledge_graph_version": "herbal-recommendation-v1",
            "partial_enrichment": True,
        }),
    )

    response = client.post("/api/herbal-recommendations/analyze", json={"complaint": "batuk ringan"})
    assert response.status_code == 200
    assert response.json()["status"] == "completed_with_partial_enrichment"

