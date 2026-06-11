// 010_herbal_recommendation_schema.cypher
// Idempotent schema marker and graph version for verified herbal recommendations.
// No recommendation facts are seeded here.

MERGE (m:SchemaMigration {id: "010_herbal_recommendation_schema"})
ON CREATE SET m.executedAt = datetime(), m.description = "Create verified herbal recommendation schema labels and version"
ON MATCH SET m.lastCheckedAt = datetime();

MERGE (v:KnowledgeGraphVersion {version: "herbal-recommendation-v1"})
ON CREATE SET v.releasedAt = datetime(), v.status = "active"
ON MATCH SET v.status = coalesce(v.status, "active"), v.lastCheckedAt = datetime();

// Label anchors make the intended schema visible without fake content.
MERGE (:PreparationMethod {id: "__schema_anchor_preparation_method__"})
MERGE (:UsageRule {id: "__schema_anchor_usage_rule__"})
MERGE (:Contraindication {id: "__schema_anchor_contraindication__"})
MERGE (:DrugInteraction {id: "__schema_anchor_drug_interaction__"})
MERGE (:SideEffect {id: "__schema_anchor_side_effect__"})
MERGE (:RiskGroup {id: "__schema_anchor_risk_group__"})
MERGE (:Warning {id: "__schema_anchor_warning__"})
MERGE (:AvailabilityProfile {id: "__schema_anchor_availability_profile__"})
MERGE (:EvidenceClaim {id: "__schema_anchor_evidence_claim__"})
MERGE (:PlantPart {id: "__schema_anchor_plant_part__"})
MERGE (:DosageForm {id: "__schema_anchor_dosage_form__"});
