// 011_herbal_recommendation_constraints.cypher
// Idempotent constraints and indexes for verified herbal recommendation graph.

MERGE (m:SchemaMigration {id: "011_herbal_recommendation_constraints"})
ON CREATE SET m.executedAt = datetime(), m.description = "Create constraints and indexes for verified herbal recommendation graph"
ON MATCH SET m.lastCheckedAt = datetime();

CREATE CONSTRAINT preparation_method_id_unique IF NOT EXISTS
FOR (n:PreparationMethod)
REQUIRE n.id IS UNIQUE;

CREATE CONSTRAINT usage_rule_id_unique IF NOT EXISTS
FOR (n:UsageRule)
REQUIRE n.id IS UNIQUE;

CREATE CONSTRAINT contraindication_id_unique IF NOT EXISTS
FOR (n:Contraindication)
REQUIRE n.id IS UNIQUE;

CREATE CONSTRAINT interaction_id_unique IF NOT EXISTS
FOR (n:DrugInteraction)
REQUIRE n.id IS UNIQUE;

CREATE CONSTRAINT side_effect_id_unique IF NOT EXISTS
FOR (n:SideEffect)
REQUIRE n.id IS UNIQUE;

CREATE CONSTRAINT risk_group_id_unique IF NOT EXISTS
FOR (n:RiskGroup)
REQUIRE n.id IS UNIQUE;

CREATE CONSTRAINT warning_id_unique IF NOT EXISTS
FOR (n:Warning)
REQUIRE n.id IS UNIQUE;

CREATE CONSTRAINT availability_profile_id_unique IF NOT EXISTS
FOR (n:AvailabilityProfile)
REQUIRE n.id IS UNIQUE;

CREATE CONSTRAINT source_id_unique IF NOT EXISTS
FOR (n:Source)
REQUIRE n.id IS UNIQUE;

CREATE CONSTRAINT evidence_claim_id_unique IF NOT EXISTS
FOR (n:EvidenceClaim)
REQUIRE n.id IS UNIQUE;

CREATE CONSTRAINT plant_part_id_unique IF NOT EXISTS
FOR (n:PlantPart)
REQUIRE n.id IS UNIQUE;

CREATE CONSTRAINT dosage_form_id_unique IF NOT EXISTS
FOR (n:DosageForm)
REQUIRE n.id IS UNIQUE;

CREATE INDEX herb_canonical_scientific_name IF NOT EXISTS
FOR (h:Herb)
ON (h.canonicalScientificName);

CREATE INDEX therapeutic_use_normalized_name IF NOT EXISTS
FOR (u:TherapeuticUse)
ON (u.normalizedName);

CREATE INDEX source_identifier IF NOT EXISTS
FOR (s:Source)
ON (s.identifier);

CREATE INDEX availability_country_code IF NOT EXISTS
FOR (a:AvailabilityProfile)
ON (a.countryCode);
