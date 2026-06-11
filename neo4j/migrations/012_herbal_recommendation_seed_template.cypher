// 012_herbal_recommendation_seed_template.cypher
// This migration intentionally does not seed recommendation facts.
// Verified herbal recommendation facts must be imported from curated JSON with sources.

MERGE (m:SchemaMigration {id: "012_herbal_recommendation_seed_template"})
ON CREATE SET m.executedAt = datetime(), m.description = "No-op seed placeholder; use scripts/import_verified_herbal_recommendation_data.py"
ON MATCH SET m.lastCheckedAt = datetime();

MERGE (v:KnowledgeGraphVersion {version: "herbal-recommendation-v1"})
ON CREATE SET v.releasedAt = datetime(), v.status = "active"
ON MATCH SET v.status = coalesce(v.status, "active"), v.lastCheckedAt = datetime();
