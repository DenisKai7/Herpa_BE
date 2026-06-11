"""Safe, idempotent Neo4j Herb duplicate-node migration helpers.

Dry-run is the default. Destructive deletion requires ``delete_duplicates=True``.
"""

import logging
import re
from typing import Any, Dict, List

from app.core.database import neo4j_driver

logger = logging.getLogger(__name__)


def normalize_scientific_name(value: str | None) -> str:
    if not value:
        return ""
    text = value.casefold()
    text = re.sub(r"[.,()]", " ", text)
    return " ".join(text.split())


def get_duplicate_groups() -> List[Dict[str, Any]]:
    cypher = """
    MATCH (h:Herb)
    WITH toLower(trim(coalesce(h.canonicalScientificName, h.latinName, ""))) AS normalized_latin, h
    WHERE normalized_latin <> ""
    WITH normalized_latin,
         collect({
             element_id: elementId(h),
             id: h.id,
             commonName: h.commonName,
             latinName: h.latinName,
             speciesNumber: h.speciesNumber,
             properties: properties(h)
         }) AS nodes,
         count(*) AS total
    WHERE total > 1
    RETURN normalized_latin, nodes, total
    ORDER BY total DESC
    """
    records, _, _ = neo4j_driver.execute_query(cypher)
    return [record.data() for record in records]


def select_canonical_node(nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    def sort_key(n: Dict[str, Any]) -> Any:
        try:
            species_number = int(n.get("speciesNumber") or 999999)
        except (TypeError, ValueError):
            species_number = 999999
        return (species_number, str(n.get("id") or "zzzzz"), str(n.get("element_id") or "zzzzz"))

    return sorted(nodes, key=sort_key)[0]


def _safe_relationship_type(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
        raise ValueError(f"Unsafe relationship type: {value}")
    return value


def migrate_single_duplicate(
    canonical: Dict[str, Any],
    duplicate: Dict[str, Any],
    *,
    dry_run: bool = True,
    delete_duplicate: bool = False,
) -> Dict[str, Any]:
    c_el_id = canonical["element_id"]
    d_el_id = duplicate["element_id"]
    d_db_id = duplicate.get("id")

    out_records, _, _ = neo4j_driver.execute_query(
        """
        MATCH (d:Herb)-[r]->(target)
        WHERE elementId(d) = $dup_el_id
        RETURN type(r) AS type, elementId(target) AS target_id, properties(r) AS props
        """,
        dup_el_id=d_el_id,
    )
    in_records, _, _ = neo4j_driver.execute_query(
        """
        MATCH (source)-[r]->(d:Herb)
        WHERE elementId(d) = $dup_el_id
        RETURN type(r) AS type, elementId(source) AS source_id, properties(r) AS props
        """,
        dup_el_id=d_el_id,
    )
    planned = {"outgoing": len(out_records), "incoming": len(in_records), "deleted": False, "dry_run": dry_run}
    if dry_run:
        return planned

    for rel in out_records:
        rel_type = _safe_relationship_type(rel["type"])
        neo4j_driver.execute_query(
            f"""
            MATCH (c:Herb), (t)
            WHERE elementId(c) = $c_el_id AND elementId(t) = $target_id
            MERGE (c)-[r:`{rel_type}`]->(t)
            ON CREATE SET r = $props
            ON MATCH SET r += $props
            """,
            c_el_id=c_el_id,
            target_id=rel["target_id"],
            props=rel["props"],
        )

    for rel in in_records:
        rel_type = _safe_relationship_type(rel["type"])
        neo4j_driver.execute_query(
            f"""
            MATCH (s), (c:Herb)
            WHERE elementId(s) = $source_id AND elementId(c) = $c_el_id
            MERGE (s)-[r:`{rel_type}`]->(c)
            ON CREATE SET r = $props
            ON MATCH SET r += $props
            """,
            source_id=rel["source_id"],
            c_el_id=c_el_id,
            props=rel["props"],
        )

    neo4j_driver.execute_query(
        """
        MATCH (c:Herb)
        WHERE elementId(c) = $c_el_id
        SET c.mergedHerbIds = [x IN coalesce(c.mergedHerbIds, []) + $dup_db_id WHERE x IS NOT NULL]
        """,
        c_el_id=c_el_id,
        dup_db_id=d_db_id,
    )

    for key, value in (duplicate.get("properties") or {}).items():
        if value is not None and key not in {"id", "element_id", "speciesNumber", "canonicalScientificName"}:
            safe_key = _safe_relationship_type(key)
            neo4j_driver.execute_query(
                f"""
                MATCH (c:Herb)
                WHERE elementId(c) = $c_el_id AND c.`{safe_key}` IS NULL
                SET c.`{safe_key}` = $value
                """,
                c_el_id=c_el_id,
                value=value,
            )

    if delete_duplicate:
        neo4j_driver.execute_query(
            """
            MATCH (d:Herb)
            WHERE elementId(d) = $d_el_id
            DETACH DELETE d
            """,
            d_el_id=d_el_id,
        )
        planned["deleted"] = True
    return planned


def run_herb_deduplication_migration(*, dry_run: bool = True, delete_duplicates: bool = False) -> Dict[str, Any]:
    groups = get_duplicate_groups()
    total_duplicates = 0
    migrated = 0
    plans = []
    for group in groups:
        canonical = select_canonical_node(group["nodes"])
        duplicates = [n for n in group["nodes"] if n["element_id"] != canonical["element_id"]]
        total_duplicates += len(duplicates)
        for duplicate in duplicates:
            plan = migrate_single_duplicate(canonical, duplicate, dry_run=dry_run, delete_duplicate=delete_duplicates)
            plans.append({"normalized_latin": group["normalized_latin"], "canonical_id": canonical.get("id"), "duplicate_id": duplicate.get("id"), **plan})
            migrated += 0 if dry_run else 1

    if not dry_run and delete_duplicates:
        neo4j_driver.execute_query(
            """
            MATCH (h:Herb)
            WHERE h.latinName IS NOT NULL AND h.canonicalScientificName IS NULL
            SET h.canonicalScientificName = toLower(trim(h.latinName))
            """
        )
        neo4j_driver.execute_query(
            """
            CREATE CONSTRAINT herb_canonical_latin_name_unique IF NOT EXISTS
            FOR (h:Herb)
            REQUIRE h.canonicalScientificName IS UNIQUE
            """
        )

    return {
        "status": "dry_run" if dry_run else "completed",
        "total_groups_processed": len(groups),
        "total_duplicates_found": total_duplicates,
        "total_duplicates_migrated": migrated,
        "delete_duplicates": delete_duplicates,
        "plans": plans,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Migration Result:", run_herb_deduplication_migration())
