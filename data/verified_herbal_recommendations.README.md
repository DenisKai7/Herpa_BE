# Verified Herbal Recommendation Dataset

`verified_herbal_recommendations.template.json` is intentionally empty. This repository currently does not contain curated production herbal facts. Do not fill recommendation facts from LLM output.

## Production eligibility

A herb can be displayed only when every required fact exists, is verified, and is connected to an active `Source` through `VERIFIED_BY`:

- identity: `Herb.commonName`, `Herb.latinName`/`canonicalScientificName`
- therapeutic use: `(:Herb)-[:USED_FOR]->(:TherapeuticUse)`
- preparation: `(:Herb)-[:HAS_PREPARATION]->(:PreparationMethod)`
- usage rule: `(:Herb)-[:HAS_USAGE_RULE]->(:UsageRule)`
- contraindication evaluation: `(:Herb)-[:HAS_CONTRAINDICATION]->(:Contraindication)`
- interaction evaluation: `(:Herb)-[:HAS_INTERACTION]->(:DrugInteraction)`
- side effect evaluation: `(:Herb)-[:HAS_SIDE_EFFECT]->(:SideEffect)`
- risk group evaluation: `(:Herb)-[:HAS_RISK_GROUP]->(:RiskGroup)`
- warning/stop-use facts: `(:Herb)-[:HAS_WARNING]->(:Warning)`
- availability: `(:Herb)-[:HAS_AVAILABILITY]->(:AvailabilityProfile)`
- provenance: every fact node has `(:FactNode)-[:VERIFIED_BY]->(:Source {active: true})`

## Source rules

- Do not use LLM output as a source.
- Every displayed field must have `source_ids`.
- `UsageRule`, contraindication, drug interaction, side effect, risk group, and warning records require source quality grade `A` or `B`.
- Traditional preparation may use grade `C` only when clearly labeled traditional.
- Every imported node must use `verification_status: "verified"` to be eligible for production.
- Empty safety arrays do not mean safe. Use explicit sourced records with `status: "no_known_issue_within_source_scope"` when the source scope supports that statement.
- Candidate records with `status: "missing"` or `status: "conflicting"` are not production eligible.

## Required preparation fields

Each `preparation_methods` item must include:

- `id`
- `title`
- `plant_part`
- `dosage_form`
- `ingredients` with `name`, optional `amount_text`, and `source_ids`
- ordered `steps`
- `suitable_symptoms`
- `evidence_level`
- `verification_status`
- `source_ids`

Only include water volume, temperature, preparation duration, storage, and extra materials when they are source-backed. Do not add honey, sugar, milk, lemon, or other ingredients unless the source states them.

## Required usage fields

Each `usage_rules` item must include:

- `id`
- `amount_text`
- `frequency_text`
- `duration_text`
- optional source-backed `administration_time_text`, `maximum_duration_text`, `before_or_after_meal`
- `allowed_age_groups`
- `prohibited_age_groups`
- `evidence_level`
- `verification_status`
- `source_ids`

Do not create numeric doses from model knowledge.

## Import

Dry-run first:

```bash
python scripts/import_verified_herbal_recommendation_data.py --input data/verified_herbal_recommendations.template.json --dry-run
```

Apply only after curation review:

```bash
python scripts/import_verified_herbal_recommendation_data.py --input data/verified_herbal_recommendations.json --apply
```
