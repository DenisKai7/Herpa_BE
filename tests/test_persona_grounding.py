import os
import sys

import pytest

sys.path.append(os.getcwd())
os.environ["SUPABASE_URL"] = "https://example.supabase.co"
os.environ["SUPABASE_SERVICE_KEY"] = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJyb2xlIjoic2VydmljZV9yb2xlIiwiaXNzIjoic3VwYWJhc2UifQ.fake-signature"
os.environ["HF_API_TOKEN"] = "dummyhftoken"

from app.agent import orchestrator
from app.agent.plant_identity import (
    CanonicalPlantIdentity,
    build_grounded_context,
    filter_records_by_entity_lock,
    resolve_canonical_plant_identity,
)
from app.agent.validators import validate_identity_consistency
from app.core.dependencies import ModelTier, Persona

QUERY = "senyawa aktif di dalam daun kelor dan kegunaannya apa aja?"
WRONG_SPECIES = ["Pyrrosia", "Graptophyllum pictum"]


def _moringa_identity() -> CanonicalPlantIdentity:
    return resolve_canonical_plant_identity(QUERY)


def _records():
    return [
        {
            "entity_id": "moringa-1",
            "local_name": "kelor",
            "scientific_name": "Moringa oleifera",
            "compounds": ["quercetin", "kaempferol", "chlorogenic acid"],
            "traditional_uses": ["pangan", "penggunaan tradisional sebagai antioksidan"],
            "evidence_level": "phytochemical_screening",
        },
        {
            "entity_id": "pyrrosia-1",
            "local_name": "sisik naga",
            "scientific_name": "Pyrrosia petiolosa",
            "compounds": ["wrong compound"],
        },
        {
            "entity_id": "daun-ungu-1",
            "local_name": "daun ungu",
            "scientific_name": "Graptophyllum pictum",
            "compounds": ["wrong compound 2"],
        },
    ]


def test_kelor_resolves_to_moringa_oleifera():
    identity = resolve_canonical_plant_identity(QUERY)

    assert identity.canonical_local_name == "kelor"
    assert identity.scientific_name == "Moringa oleifera"
    assert identity.confidence >= 0.9
    assert identity.resolution_method == "exact_alias_match"


def test_contamination_filter_accepts_only_moringa():
    identity = _moringa_identity()
    accepted, general, rejected = filter_records_by_entity_lock(_records(), identity)

    assert [r["scientific_name"] for r in accepted] == ["Moringa oleifera"]
    assert general == []
    assert {r["scientific_name"] for r in rejected} == {"Pyrrosia petiolosa", "Graptophyllum pictum"}


def test_grounded_context_metadata_counts_rejected_records():
    identity = _moringa_identity()
    context = build_grounded_context(
        query=QUERY,
        identity=identity,
        vector_records=_records(),
        graph_records=[],
        persona=Persona.PENELITI,
        tier=ModelTier.THINKING,
    )

    assert context.retrieval_metadata["accepted_count"] == 1
    assert context.retrieval_metadata["rejected_entity_mismatch"] == 2
    prompt_context = context.to_prompt_text()
    assert "Moringa oleifera" in prompt_context
    assert "CONFLICTING RECORDS REMOVED" in prompt_context


def test_validator_rejects_wrong_species_for_kelor():
    identity = _moringa_identity()
    answer = "Daun kelor adalah Moringa oleifera, bukan Pyrrosia petiolosa."
    result = validate_identity_consistency(answer, identity)

    assert result.passed is False
    assert any("wrong_species_mentioned" in reason for reason in result.reasons)


def test_unknown_plant_not_substituted():
    identity = resolve_canonical_plant_identity("senyawa aktif daun tanamanxyz apa?")

    assert identity.resolution_method == "not_found"
    assert identity.scientific_name is None


@pytest.mark.parametrize("persona", ["umum", "pelajar", "peneliti", "tenaga_medis"])
def test_fast_and_thinking_keep_same_identity_and_different_depth(monkeypatch, persona):
    identity = _moringa_identity()
    context = build_grounded_context(
        query=QUERY,
        identity=identity,
        vector_records=[_records()[0]],
        graph_records=[],
        persona=Persona(persona),
        tier=ModelTier.THINKING,
    )

    def fake_retrieve_grounded_context(**kwargs):
        tier = kwargs["tier"]
        return build_grounded_context(
            query=QUERY,
            identity=identity,
            vector_records=[_records()[0]],
            graph_records=[],
            persona=Persona(persona),
            tier=tier,
        )

    def fake_generate(**kwargs):
        tier = kwargs["model_tier"]
        base = (
            "Identitas tanaman: kelor (Moringa oleifera). "
            "Senyawa utama: quercetin, kaempferol, dan chlorogenic acid. "
            "Kegunaan bersifat potensial/tradisional dan bukti klinis masih terbatas. "
            "Peringatan: tidak menggantikan obat dokter."
        )
        if tier == "thinking":
            return base + (
                "\n\nEvidence level: phytochemical_screening untuk kandungan, in_vitro untuk sebagian aktivitas antioksidan. "
                "Limitations: data klinis manusia masih terbatas. Safety: perhatikan interaksi obat, kehamilan, penyakit kronis. "
                "Research gap: standardisasi ekstrak, dosis aman, dan uji klinis terkontrol masih diperlukan."
            )
        return base

    monkeypatch.setattr(orchestrator, "retrieve_grounded_context", fake_retrieve_grounded_context)
    monkeypatch.setattr(orchestrator, "generate_strict_response", fake_generate)

    fast = orchestrator._process_query_sync(QUERY, persona, model_tier="fast")
    thinking = orchestrator._process_query_sync(QUERY, persona, model_tier="thinking")

    assert fast["metadata"]["canonical_entity"]["scientific_name"] == "Moringa oleifera"
    assert thinking["metadata"]["canonical_entity"]["scientific_name"] == "Moringa oleifera"
    assert len(fast["ai_response"].split()) < len(thinking["ai_response"].split())
    assert "quercetin" in fast["ai_response"]
    assert "quercetin" in thinking["ai_response"]
    assert "Evidence level" in thinking["ai_response"]
    for wrong in WRONG_SPECIES:
        assert wrong not in fast["ai_response"]
        assert wrong not in thinking["ai_response"]


def test_persona_specific_content_with_mock_generation(monkeypatch):
    expected = {
        "umum": "Cara pengolahan sederhana dan peringatan untuk obat rutin.",
        "pelajar": "Quercetin termasuk flavonoid, yaitu metabolit sekunder tanaman.",
        "peneliti": "Formula kimia belum tersedia pada sumber retrieval; evidence level phytochemical_screening.",
        "tenaga_medis": "Kontraindikasi, interaksi obat-herbal, efek samping, dan monitoring perlu diperhatikan.",
    }

    def fake_retrieve_grounded_context(**kwargs):
        return build_grounded_context(
            query=QUERY,
            identity=_moringa_identity(),
            vector_records=[_records()[0]],
            graph_records=[],
            persona=Persona(kwargs.get("persona", "umum")),
            tier=kwargs["tier"],
        )

    def fake_generate(**kwargs):
        persona = kwargs["ai_mode"]
        return f"Kelor (Moringa oleifera). {expected[persona]} Bukti klinis masih terbatas."

    monkeypatch.setattr(orchestrator, "retrieve_grounded_context", fake_retrieve_grounded_context)
    monkeypatch.setattr(orchestrator, "generate_strict_response", fake_generate)

    for persona, marker in expected.items():
        result = orchestrator._process_query_sync(QUERY, persona, model_tier="fast")
        assert marker in result["ai_response"]
        assert result["metadata"]["validation"]["identity_consistent"] is True
        assert result["metadata"]["validation"]["claims_grounded"] is True


def test_orchestrator_safe_fallback_after_invalid_generation(monkeypatch):
    def fake_retrieve_grounded_context(**kwargs):
        return build_grounded_context(
            query=QUERY,
            identity=_moringa_identity(),
            vector_records=[_records()[0]],
            graph_records=[],
            persona=Persona.UMUM,
            tier=kwargs["tier"],
        )

    def fake_generate(**kwargs):
        return "Daun kelor adalah Pyrrosia petiolosa dan pasti menyembuhkan semua penyakit."

    monkeypatch.setattr(orchestrator, "retrieve_grounded_context", fake_retrieve_grounded_context)
    monkeypatch.setattr(orchestrator, "generate_strict_response", fake_generate)

    result = orchestrator._process_query_sync(QUERY, "umum", model_tier="fast")

    assert "Moringa oleifera" in result["ai_response"]
    assert "Pyrrosia" not in result["ai_response"]
    assert "tidak akan mengganti konteks" in result["ai_response"]
    assert result["metadata"]["validation"]["passed"] is False
