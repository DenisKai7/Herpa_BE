import pytest
from pydantic import ValidationError

from scripts.import_verified_herbal_recommendation_data import Dataset


def safety(id_: str, source_id="SRC-A", status="no_known_issue_within_source_scope"):
    return {
        "id": id_,
        "title": "No known issue",
        "description": "No known issue within source scope.",
        "severity": "low",
        "action_text": "Ikuti sumber terverifikasi.",
        "status": status,
        "source_ids": [source_id],
        "verification_status": "verified",
    }


def base_record(source_grade="A"):
    source = {
        "id": "SRC-A",
        "title": "Verified source",
        "identifier": "SRC-A",
        "publisher": "Publisher",
        "year": 2024,
        "qualityGrade": source_grade,
        "active": True,
    }
    return {
        "data_version": "herbal-recommendation-v1",
        "records": [{
            "herb": {"canonical_scientific_name": "kaempferia galanga", "common_name": "Kencur", "aliases": ["cekur"], "active_compounds": []},
            "therapeutic_uses": [{"id": "USE-1", "normalized_name": "batuk", "evidence_level": "traditional", "source_ids": ["SRC-A"], "verification_status": "verified"}],
            "preparation_methods": [{
                "id": "PREP-1",
                "title": "Seduhan",
                "plant_part": "rimpang",
                "dosage_form": "seduhan",
                "ingredients": [{"name": "rimpang", "amount_text": "sesuai monografi", "source_ids": ["SRC-A"]}],
                "steps": ["Cuci rimpang.", "Potong rimpang.", "Seduh sesuai sumber.", "Saring."],
                "suitable_symptoms": ["batuk"],
                "source_ids": ["SRC-A"],
                "verification_status": "verified",
            }],
            "usage_rules": [{"id": "USE-RULE-1", "amount_text": "100 ml", "frequency_text": "1 kali sehari", "duration_text": "1 hari", "allowed_age_groups": ["adult"], "prohibited_age_groups": ["infant"], "source_ids": ["SRC-A"], "verification_status": "verified"}],
            "contraindications": [safety("SAFE-1")],
            "interactions": [safety("INT-1")],
            "side_effects": [safety("SIDE-1")],
            "risk_groups": [safety("RISK-1")],
            "warnings": [safety("WARN-1", status="known_issue")],
            "stop_use_signs": ["muncul reaksi alergi"],
            "availability": {"id": "AV-1", "country_code": "ID", "category": "easy_to_find", "score": 0.8, "reason": "curated", "source_ids": ["SRC-A"], "verification_status": "verified"},
            "sources": [source],
        }],
    }


def test_import_dataset_accepts_grade_a_complete_record():
    dataset = Dataset.model_validate(base_record("A"))
    assert len(dataset.records) == 1


def test_import_dataset_rejects_grade_c_usage_rule():
    with pytest.raises(ValidationError):
        Dataset.model_validate(base_record("C"))


def test_import_dataset_rejects_missing_warning():
    data = base_record("A")
    data["records"][0]["warnings"] = []
    with pytest.raises(ValidationError):
        Dataset.model_validate(data)
