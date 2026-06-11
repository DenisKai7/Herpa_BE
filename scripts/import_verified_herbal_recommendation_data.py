"""Import curated, source-backed herbal recommendation facts into Neo4j.

Dry-run is default. This script rejects records that do not carry provenance.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Literal

sys.path.append(str(Path(__file__).resolve().parents[1]))

from pydantic import BaseModel, Field, ValidationError, model_validator

from app.core.database import neo4j_driver

QUALITY_GRADES = {"A", "B", "C", "D"}
A_B_ONLY = {"usage_rules", "contraindications", "interactions", "side_effects", "risk_groups", "warnings"}
SAFETY_STATUSES = {"known_issue", "no_known_issue_within_source_scope", "conflicting"}


class SourceRecord(BaseModel):
    id: str
    source_type: str = Field(alias="sourceType", default="reference")
    title: str
    publisher: str | None = None
    year: int | None = None
    identifier: str
    url: str | None = None
    access_date: str | None = Field(alias="accessDate", default=None)
    quality_grade: Literal["A", "B", "C", "D"] = Field(alias="qualityGrade")
    active: bool = True


class SourcedItem(BaseModel):
    id: str
    source_ids: list[str]
    verification_status: str = "verified"

    @model_validator(mode="after")
    def require_source_and_verified(self):
        if not self.source_ids:
            raise ValueError("source_ids is required")
        if self.verification_status != "verified":
            raise ValueError("only verified records are production-eligible")
        return self


class HerbRef(BaseModel):
    canonical_scientific_name: str
    common_name: str
    aliases: list[str] = Field(default_factory=list)
    active_compounds: list[str] = Field(default_factory=list)


class IngredientRecord(BaseModel):
    name: str
    amount_text: str | None = None
    source_ids: list[str]

    @model_validator(mode="after")
    def require_source(self):
        if not self.source_ids:
            raise ValueError("ingredient source_ids is required")
        return self


class TherapeuticUseRecord(SourcedItem):
    normalized_name: str
    evidence_level: str


class PreparationRecord(SourcedItem):
    title: str
    plant_part: str
    dosage_form: str
    ingredients: list[IngredientRecord]
    steps: list[str]
    water_volume_text: str | None = None
    temperature_text: str | None = None
    preparation_duration_text: str | None = None
    storage_instruction: str | None = None
    suitable_symptoms: list[str]
    evidence_level: str = "traditional"

    @model_validator(mode="after")
    def require_instruction_detail(self):
        if not self.ingredients:
            raise ValueError("preparation ingredients are required")
        if len([step for step in self.steps if step.strip()]) < 2:
            raise ValueError("preparation requires ordered non-generic steps")
        if not self.suitable_symptoms:
            raise ValueError("preparation suitable_symptoms is required")
        return self


class UsageRecord(SourcedItem):
    amount_text: str
    frequency_text: str
    administration_time_text: str | None = None
    duration_text: str
    maximum_duration_text: str | None = None
    before_or_after_meal: str | None = None
    administration_notes: list[str] = Field(default_factory=list)
    allowed_age_groups: list[str] = Field(default_factory=list)
    prohibited_age_groups: list[str] = Field(default_factory=list)
    evidence_level: str = "traditional"


class SafetyRecord(SourcedItem):
    title: str | None = None
    name: str | None = None
    label: str | None = None
    severity: str = "unspecified"
    description: str
    action_text: str
    status: Literal["known_issue", "no_known_issue_within_source_scope", "conflicting"] = "known_issue"
    verified_at: str | None = None

    @model_validator(mode="after")
    def require_title(self):
        if not (self.title or self.name or self.label):
            raise ValueError("safety title/name/label is required")
        return self


class AvailabilityRecord(SourcedItem):
    country_code: str
    category: Literal["easy_to_find", "moderately_available", "hard_to_find", "seasonal", "restricted"]
    score: float
    reason: str
    region: str | None = None
    cultivation_status: str | None = None
    market_availability: str | None = None


class HerbRecommendationRecord(BaseModel):
    herb: HerbRef
    therapeutic_uses: list[TherapeuticUseRecord]
    preparation_methods: list[PreparationRecord]
    usage_rules: list[UsageRecord]
    contraindications: list[SafetyRecord]
    interactions: list[SafetyRecord]
    side_effects: list[SafetyRecord]
    risk_groups: list[SafetyRecord]
    warnings: list[SafetyRecord]
    stop_use_signs: list[str]
    availability: AvailabilityRecord
    sources: list[SourceRecord]

    @model_validator(mode="after")
    def validate_source_coverage(self):
        source_map = {source.id: source for source in self.sources if source.active}
        if not source_map:
            raise ValueError("at least one active source is required")
        if not self.therapeutic_uses:
            raise ValueError("therapeutic_uses is required")
        if not self.preparation_methods:
            raise ValueError("preparation_methods is required")
        if not self.usage_rules:
            raise ValueError("usage_rules is required")
        if not self.stop_use_signs:
            raise ValueError("stop_use_signs is required")

        buckets = {
            "therapeutic_uses": self.therapeutic_uses,
            "preparation_methods": self.preparation_methods,
            "usage_rules": self.usage_rules,
            "contraindications": self.contraindications,
            "interactions": self.interactions,
            "side_effects": self.side_effects,
            "risk_groups": self.risk_groups,
            "warnings": self.warnings,
            "availability": [self.availability],
        }
        for bucket_name in ["contraindications", "interactions", "side_effects", "risk_groups", "warnings"]:
            if not buckets[bucket_name]:
                raise ValueError(f"{bucket_name} requires explicit sourced evaluation")
        for bucket_name, items in buckets.items():
            for item in items:
                missing = [source_id for source_id in item.source_ids if source_id not in source_map]
                if missing:
                    raise ValueError(f"{bucket_name} references inactive or missing source_ids: {missing}")
                if bucket_name in A_B_ONLY and any(source_map[source_id].quality_grade not in {"A", "B"} for source_id in item.source_ids):
                    raise ValueError(f"{bucket_name} requires source grade A or B")
                if bucket_name == "preparation_methods" and item.evidence_level != "traditional" and any(source_map[source_id].quality_grade not in {"A", "B"} for source_id in item.source_ids):
                    raise ValueError("non-traditional preparation requires source grade A or B")
                if isinstance(item, PreparationRecord):
                    for ingredient in item.ingredients:
                        missing_ingredient = [source_id for source_id in ingredient.source_ids if source_id not in source_map]
                        if missing_ingredient:
                            raise ValueError(f"ingredient references inactive or missing source_ids: {missing_ingredient}")
        return self


class Dataset(BaseModel):
    data_version: str = "herbal-recommendation-v1"
    records: list[HerbRecommendationRecord] = Field(default_factory=list)


def load_dataset(path: Path) -> Dataset:
    return Dataset.model_validate(json.loads(path.read_text(encoding="utf-8")))


def _dump(value: BaseModel | dict | list) -> dict | list:
    if isinstance(value, BaseModel):
        return value.model_dump(by_alias=True)
    return value


def import_record(record: HerbRecommendationRecord, data_version: str) -> None:
    params = record.model_dump(by_alias=True)
    params["data_version"] = data_version
    neo4j_driver.execute_query(
        """
        MERGE (h:Herb {canonicalScientificName: $herb.canonical_scientific_name})
        ON CREATE SET h.commonName = $herb.common_name, h.latinName = $herb.canonical_scientific_name
        SET h.commonName = $herb.common_name,
            h.latinName = $herb.canonical_scientific_name,
            h.localNames = $herb.aliases,
            h.activeCompounds = $herb.active_compounds,
            h.dataVersion = $data_version,
            h.lastVerifiedAt = datetime()
        WITH h
        UNWIND $sources AS source
        MERGE (s:Source {id: source.id})
        SET s.sourceType = source.sourceType,
            s.title = source.title,
            s.publisher = source.publisher,
            s.year = source.year,
            s.identifier = source.identifier,
            s.url = source.url,
            s.accessDate = source.accessDate,
            s.qualityGrade = source.qualityGrade,
            s.active = source.active,
            s.dataVersion = $data_version
        """,
        parameters_=params,
    )
    for key, label, rel in [
        ("therapeutic_uses", "TherapeuticUse", "USED_FOR"),
        ("preparation_methods", "PreparationMethod", "HAS_PREPARATION"),
        ("usage_rules", "UsageRule", "HAS_USAGE_RULE"),
        ("contraindications", "Contraindication", "HAS_CONTRAINDICATION"),
        ("interactions", "DrugInteraction", "HAS_INTERACTION"),
        ("side_effects", "SideEffect", "HAS_SIDE_EFFECT"),
        ("risk_groups", "RiskGroup", "HAS_RISK_GROUP"),
        ("warnings", "Warning", "HAS_WARNING"),
    ]:
        neo4j_driver.execute_query(
            f"""
            MATCH (h:Herb {{canonicalScientificName: $herb.canonical_scientific_name}})
            UNWIND ${key} AS item
            MERGE (n:{label} {{id: item.id}})
            SET n += item,
                n.dataVersion = $data_version,
                n.lastVerifiedAt = datetime(),
                n.verificationStatus = item.verification_status,
                n.stopUseSigns = CASE WHEN '{label}' = 'Warning' THEN $stop_use_signs ELSE n.stopUseSigns END
            MERGE (h)-[:{rel}]->(n)
            WITH n, item
            UNWIND item.source_ids AS sourceId
            MATCH (s:Source {{id: sourceId}})
            MERGE (n)-[:VERIFIED_BY]->(s)
            """,
            parameters_=params,
        )
    neo4j_driver.execute_query(
        """
        MATCH (h:Herb {canonicalScientificName: $herb.canonical_scientific_name})
        MERGE (a:AvailabilityProfile {id: $availability.id})
        SET a += $availability,
            a.dataVersion = $data_version,
            a.lastVerifiedAt = datetime(),
            a.verificationStatus = $availability.verification_status
        MERGE (h)-[:HAS_AVAILABILITY]->(a)
        WITH a
        UNWIND $availability.source_ids AS sourceId
        MATCH (s:Source {id: sourceId})
        MERGE (a)-[:VERIFIED_BY]->(s)
        """,
        parameters_=params,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if args.dry_run and args.apply:
        parser.error("choose either --dry-run or --apply")
    path = Path(args.input)
    try:
        dataset = load_dataset(path)
    except (ValidationError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "rejected", "error": str(exc)}, ensure_ascii=False, indent=2))
        return 1
    report = {"status": "dry_run" if not args.apply else "applied", "accepted_records": len(dataset.records), "rejected_records": 0, "data_version": dataset.data_version}
    if args.apply:
        for record in dataset.records:
            import_record(record, dataset.data_version)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
