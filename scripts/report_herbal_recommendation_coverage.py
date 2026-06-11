"""Report verified herbal recommendation coverage from Neo4j."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.core.database import neo4j_driver

FIELDS = ["Use", "Preparation", "Usage", "Contraindication", "Interaction", "SideEffect", "RiskGroup", "Availability", "Source"]

COVERAGE_QUERY = """
MATCH (h:Herb)
OPTIONAL MATCH (h)-[:USED_FOR]->(use:TherapeuticUse)
OPTIONAL MATCH (use)-[:VERIFIED_BY]->(useSource:Source)
OPTIONAL MATCH (h)-[:HAS_PREPARATION]->(prep:PreparationMethod)-[:VERIFIED_BY]->(prepSource:Source)
OPTIONAL MATCH (h)-[:HAS_USAGE_RULE]->(usage:UsageRule)-[:VERIFIED_BY]->(usageSource:Source)
OPTIONAL MATCH (h)-[:HAS_CONTRAINDICATION]->(contra:Contraindication)-[:VERIFIED_BY]->(contraSource:Source)
OPTIONAL MATCH (h)-[:HAS_INTERACTION]->(interaction:DrugInteraction)-[:VERIFIED_BY]->(interactionSource:Source)
OPTIONAL MATCH (h)-[:HAS_SIDE_EFFECT]->(side:SideEffect)-[:VERIFIED_BY]->(sideSource:Source)
OPTIONAL MATCH (h)-[:HAS_RISK_GROUP]->(risk:RiskGroup)-[:VERIFIED_BY]->(riskSource:Source)
OPTIONAL MATCH (h)-[:HAS_AVAILABILITY]->(availability:AvailabilityProfile)-[:VERIFIED_BY]->(availabilitySource:Source)
WITH h, useSource, prep, prepSource, usage, usageSource, contra, contraSource, interaction, interactionSource, side, sideSource, risk, riskSource, availability, availabilitySource
WITH h,
     count(DISTINCT CASE WHEN coalesce(properties(useSource)["active"], true) = true THEN useSource END) > 0 AS has_use,
     count(DISTINCT CASE WHEN properties(prep)["verificationStatus"] = 'verified' AND coalesce(properties(prepSource)["active"], true) = true THEN prepSource END) > 0 AS has_preparation,
     count(DISTINCT CASE WHEN properties(usage)["verificationStatus"] = 'verified' AND coalesce(properties(usageSource)["active"], true) = true THEN usageSource END) > 0 AS has_usage,
     count(DISTINCT CASE WHEN properties(contra)["verificationStatus"] = 'verified' AND coalesce(properties(contraSource)["active"], true) = true THEN contraSource END) > 0 AS has_contraindication,
     count(DISTINCT CASE WHEN properties(interaction)["verificationStatus"] = 'verified' AND coalesce(properties(interactionSource)["active"], true) = true THEN interactionSource END) > 0 AS has_interaction,
     count(DISTINCT CASE WHEN properties(side)["verificationStatus"] = 'verified' AND coalesce(properties(sideSource)["active"], true) = true THEN sideSource END) > 0 AS has_side_effect,
     count(DISTINCT CASE WHEN properties(risk)["verificationStatus"] = 'verified' AND coalesce(properties(riskSource)["active"], true) = true THEN riskSource END) > 0 AS has_risk_group,
     count(DISTINCT CASE WHEN properties(availability)["verificationStatus"] = 'verified' AND coalesce(properties(availabilitySource)["active"], true) = true THEN availabilitySource END) > 0 AS has_availability,
     count(DISTINCT useSource) + count(DISTINCT prepSource) + count(DISTINCT usageSource) + count(DISTINCT contraSource) + count(DISTINCT interactionSource) + count(DISTINCT sideSource) + count(DISTINCT riskSource) + count(DISTINCT availabilitySource) AS source_count
WITH h, properties(h) AS hp, has_use, has_preparation, has_usage, has_contraindication, has_interaction, has_side_effect, has_risk_group, has_availability, source_count
RETURN hp["commonName"] AS herb,
       hp["canonicalScientificName"] AS canonicalScientificName,
       has_use, has_preparation, has_usage, has_contraindication, has_interaction,
       has_side_effect, has_risk_group, has_availability,
       source_count > 0 AS has_source
ORDER BY herb ASC
"""


def collect_rows() -> list[dict]:
    rel_records, _, _ = neo4j_driver.execute_query("CALL db.relationshipTypes() YIELD relationshipType RETURN collect(relationshipType) AS rels")
    rels = set(rel_records[0].data().get("rels", [])) if rel_records else set()
    required_rels = {"HAS_PREPARATION", "HAS_USAGE_RULE", "HAS_CONTRAINDICATION", "HAS_INTERACTION", "HAS_SIDE_EFFECT", "HAS_RISK_GROUP", "HAS_AVAILABILITY"}
    if not required_rels.issubset(rels):
        records, _, _ = neo4j_driver.execute_query('MATCH (h:Herb) WITH properties(h) AS hp RETURN hp["commonName"] AS herb, hp["canonicalScientificName"] AS canonicalScientificName ORDER BY herb ASC')
        return [
            {
                "Herb": record.data().get("herb"),
                "CanonicalScientificName": record.data().get("canonicalScientificName"),
                "Identity": 1 if record.data().get("herb") and record.data().get("canonicalScientificName") else 0,
                **{key: 0 for key in FIELDS},
                "Coverage": 0.1 if record.data().get("herb") and record.data().get("canonicalScientificName") else 0.0,
                "MissingFields": ";".join(FIELDS),
            }
            for record in records
        ]
    records, _, _ = neo4j_driver.execute_query(COVERAGE_QUERY)
    rows = []
    for record in records:
        data = record.data()
        checks = {
            "Use": data["has_use"],
            "Preparation": data["has_preparation"],
            "Usage": data["has_usage"],
            "Contraindication": data["has_contraindication"],
            "Interaction": data["has_interaction"],
            "SideEffect": data["has_side_effect"],
            "RiskGroup": data["has_risk_group"],
            "Availability": data["has_availability"],
            "Source": data["has_source"],
        }
        missing = [key for key, value in checks.items() if not value]
        rows.append({
            "Herb": data.get("herb"),
            "CanonicalScientificName": data.get("canonicalScientificName"),
            "Identity": 1 if data.get("herb") and data.get("canonicalScientificName") else 0,
            **{key: 1 if value else 0 for key, value in checks.items()},
            "Coverage": round((sum(1 for value in checks.values() if value) + (1 if data.get("herb") and data.get("canonicalScientificName") else 0)) / 10, 4),
            "MissingFields": ";".join(missing),
        })
    return rows


def summarize(rows: list[dict]) -> dict:
    total = len(rows)
    fully = sum(1 for row in rows if row["Coverage"] == 1.0)
    average = round(sum(row["Coverage"] for row in rows) / total, 4) if total else 0.0
    return {
        "total_herbs": total,
        "fully_verified_herbs": fully,
        "partially_verified_herbs": total - fully,
        "average_coverage": average,
        "missing_preparation_count": sum(1 for row in rows if row["Preparation"] == 0),
        "missing_usage_count": sum(1 for row in rows if row["Usage"] == 0),
        "missing_safety_count": sum(1 for row in rows if row["Contraindication"] == 0 or row["Interaction"] == 0 or row["SideEffect"] == 0 or row["RiskGroup"] == 0),
        "missing_availability_count": sum(1 for row in rows if row["Availability"] == 0),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    args = parser.parse_args()
    rows = collect_rows()
    if args.output:
        path = Path(args.output)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["Herb", "Coverage"])
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps({"summary": summarize(rows), "rows": rows[:20]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
