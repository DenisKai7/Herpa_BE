"""
GraphRAG Retriever - Hybrid search menggunakan Supabase pgvector + Neo4j Cypher.
"""

import logging
import re
from typing import Any, Optional

from app.core.database import neo4j_driver, supabase
from app.core.embedding import embed_text
from app.core.dependencies import Persona, PERSONA_ALIASES

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════
# HELPER: NORMALIZE PERSONA
# ═══════════════════════════════════════════
def _normalize_persona(ai_mode: str) -> Persona:
    val = str(ai_mode).lower().strip()
    return PERSONA_ALIASES.get(val, Persona.UMUM)

# ═══════════════════════════════════════════
# HELPER: VECTOR SEARCH (Supabase pgvector)
# ═══════════════════════════════════════════
def _vector_search(
    query: str,
    table: str,
    match_function: str,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Melakukan pencarian semantik menggunakan Supabase RPC."""
    try:
        query_embedding = embed_text(query)

        result = supabase.rpc(match_function, {
            "query_embedding": query_embedding,
            "match_count": limit,
            "match_threshold": 0.5,
        }).execute()

        if result.data:
            logger.info(
                f"Vector search [{match_function}]: "
                f"{len(result.data)} results found."
            )
            return result.data
        return []

    except Exception as e:
        logger.error(
            f"Vector search error [{match_function}]: {e}",
            exc_info=True,
        )
        return []

# ═══════════════════════════════════════════
# HELPER: GRAPH SEARCH (Neo4j Cypher)
# ═══════════════════════════════════════════
def _graph_search(query: str, cypher_query: str, **kwargs: Any) -> list[dict[str, Any]]:
    """Menjalankan Cypher query di Neo4j."""
    try:
        params = {"query": query}
        params.update(kwargs)
        records, _, _ = neo4j_driver.execute_query(
            cypher_query,
            parameters_=params,
        )
        result = [record.data() for record in records]
        logger.info(f"Graph search: {len(result)} records found.")
        return result
    except Exception as e:
        logger.exception(
            "Graph retrieval failed; continuing without graph context: %s",
            e,
        )
        return []

def _format_records_to_text(
    records: list[dict[str, Any]],
    source_label: str = "Database",
) -> str:
    """Mengubah list of dict menjadi teks terstruktur untuk dikirim ke LLM."""
    if not records:
        return f"[{source_label}]: Tidak ada data ditemukan."

    lines: list[str] = [f"[{source_label} - {len(records)} hasil]:"]
    for i, rec in enumerate(records, 1):
        parts: list[str] = []
        for key, value in rec.items():
            if value is not None and key not in ("embedding", "id"):
                parts.append(f"  {key}: {value}")
        lines.append(f"\n--- Hasil #{i} ---")
        lines.extend(parts)
    return "\n".join(lines)

# ═══════════════════════════════════════════
# INTENT: KONSULTASI (Content-Based Recommendation)
# ═══════════════════════════════════════════
def content_based_recommendation(query: str, limit: int = 5, graph_limit: int = 4, persona: str = "umum") -> str:
    """Hybrid search untuk intent 'konsultasi' dengan profil retrieval per-persona."""
    # ── STEP 1: Vector Search ──
    vector_results = _vector_search(query, "plants", "match_plants", limit)
    vector_text = _format_records_to_text(
        vector_results, "Pencarian Semantik Tanaman Obat"
    )

    # ── STEP 2: Graph Search ──
    tag_match = re.search(r'\[target:\s*([^\]]+)\]', query.lower() if query else "")
    if tag_match:
        exact_compound = tag_match.group(1).strip()
        cleaned_words = [exact_compound]
    else:
        cleaned_words = list(set(re.findall(r'\b\w{4,}\b', query.lower() if query else "")))
        if not cleaned_words:
            cleaned_words = [query.lower()] if query else [""]

    p_enum = _normalize_persona(persona)

    # Persona-specific Neo4j Queries
    if p_enum == Persona.UMUM:
        cypher = """
        MATCH (h:Herb)
        WHERE any(w IN $words WHERE toLower(h.commonName) CONTAINS w OR toLower(h.macroscopicDesc) CONTAINS w)
           OR toLower(h.commonName) CONTAINS toLower($query)
           OR toLower(h.macroscopicDesc) CONTAINS toLower($query)
        OPTIONAL MATCH (h)-[:USED_FOR]->(t:TherapeuticUse)
        OPTIONAL MATCH (h)-[:HAS_TOXICITY]->(tox:ToxicityCategory)
        RETURN h.commonName AS tanaman,
               h.latinName AS nama_latin,
               h.macroscopicDesc AS deskripsi_umum,
               collect(DISTINCT t.name) AS manfaat_tradisional,
               collect(DISTINCT tox.name) AS tingkat_keamanan
        LIMIT $limit
        """
    elif p_enum == Persona.PELAJAR:
        cypher = """
        MATCH (h:Herb)
        WHERE any(w IN $words WHERE toLower(h.commonName) CONTAINS w OR toLower(h.macroscopicDesc) CONTAINS w)
           OR toLower(h.commonName) CONTAINS toLower($query)
           OR toLower(h.macroscopicDesc) CONTAINS toLower($query)
        OPTIONAL MATCH (h)-[:BELONGS_TO]->(f:Family)
        OPTIONAL MATCH (h)-[:HAS_COMPOUND]->(c:Compound)
        RETURN h.commonName AS tanaman,
               h.latinName AS nama_latin,
               h.simplisiaName AS nama_simplisia,
               f.name AS famili,
               collect(DISTINCT c.name) AS senyawa_aktif
        LIMIT $limit
        """
    elif p_enum == Persona.PENELITI:
        cypher = """
        MATCH (h:Herb)
        WHERE any(w IN $words WHERE toLower(h.commonName) CONTAINS w OR toLower(h.macroscopicDesc) CONTAINS w)
           OR toLower(h.commonName) CONTAINS toLower($query)
           OR toLower(h.macroscopicDesc) CONTAINS toLower($query)
        OPTIONAL MATCH (h)-[:HAS_COMPOUND]->(c:Compound)
        OPTIONAL MATCH (h)-[:HAS_COMPOUND_CLASS]->(cc:CompoundClass)
        OPTIONAL MATCH (h)-[:HAS_PROTEIN_TARGET]->(pt:ProteinTarget)
        RETURN h.commonName AS tanaman,
               h.latinName AS nama_latin,
               h.simplisiaName AS simplisia,
               collect(DISTINCT c.name) AS senyawa_aktif,
               collect(DISTINCT cc.name) AS kelas_senyawa,
               collect(DISTINCT pt.name) AS target_protein
        LIMIT $limit
        """
    else: # Tenaga Medis
        cypher = """
        MATCH (h:Herb)
        WHERE any(w IN $words WHERE toLower(h.commonName) CONTAINS w OR toLower(h.macroscopicDesc) CONTAINS w)
           OR toLower(h.commonName) CONTAINS toLower($query)
           OR toLower(h.macroscopicDesc) CONTAINS toLower($query)
        OPTIONAL MATCH (h)-[:HAS_TOXICITY]->(tox:ToxicityCategory)
        OPTIONAL MATCH (h)-[:USED_FOR]->(t:TherapeuticUse)
        RETURN h.commonName AS tanaman,
               h.latinName AS nama_latin,
               h.simplisiaName AS simplisia,
               h.macroscopicDesc AS deskripsi_makros,
               h.microscopicDesc AS deskripsi_mikros,
               collect(DISTINCT t.name) AS indikasi_klinis,
               collect(DISTINCT tox.name) AS toksisitas_dan_efek_samping
        LIMIT $limit
        """

    graph_results = _graph_search(query, cypher, words=cleaned_words, limit=graph_limit)
    graph_text = _format_records_to_text(
        graph_results, "Relasi Graph Tanaman-Gejala-Senyawa"
    )

    return f"{vector_text}\n\n{graph_text}"

# ═══════════════════════════════════════════
# INTENT: ENSIKLOPEDIA (Encyclopedia Search)
# ═══════════════════════════════════════════
def search_encyclopedia(query: str, limit: int = 5, graph_limit: int = 4, persona: str = "umum") -> str:
    """Hybrid search untuk intent 'ensiklopedia'."""
    # ── STEP 1: Vector Search ──
    vector_results = _vector_search(
        query, "encyclopedia", "match_encyclopedia", limit
    )
    vector_text = _format_records_to_text(vector_results, "Pencarian Ensiklopedia")

    # ── STEP 2: Graph Search ──
    tag_match = re.search(r'\[target:\s*([^\]]+)\]', query.lower() if query else "")
    if tag_match:
        exact_compound = tag_match.group(1).strip()
        cleaned_words = [exact_compound]
    else:
        cleaned_words = list(set(re.findall(r'\b\w{4,}\b', query.lower() if query else "")))
        if not cleaned_words:
            cleaned_words = [query.lower()] if query else [""]

    p_enum = _normalize_persona(persona)

    # Persona-specific Neo4j Queries
    if p_enum == Persona.UMUM:
        cypher = """
        MATCH (p:Herb)
        WHERE any(w IN $words WHERE toLower(p.commonName) CONTAINS w OR toLower(p.latinName) CONTAINS w)
           OR toLower(p.commonName) CONTAINS toLower($query)
           OR toLower(p.latinName) CONTAINS toLower($query)
        OPTIONAL MATCH (p)-[:USED_FOR]->(t:TherapeuticUse)
        OPTIONAL MATCH (p)-[:HAS_TOXICITY]->(tox:ToxicityCategory)
        RETURN p.commonName AS nama, p.latinName AS nama_latin,
               p.macroscopicDesc AS deskripsi,
               collect(DISTINCT t.name) AS manfaat_tradisional,
               collect(DISTINCT tox.name) AS tingkat_keamanan
        LIMIT $limit
        """
    elif p_enum == Persona.PELAJAR:
        cypher = """
        MATCH (p:Herb)
        WHERE any(w IN $words WHERE toLower(p.commonName) CONTAINS w OR toLower(p.latinName) CONTAINS w)
           OR toLower(p.commonName) CONTAINS toLower($query)
           OR toLower(p.latinName) CONTAINS toLower($query)
        OPTIONAL MATCH (p)-[:BELONGS_TO]->(f:Family)
        OPTIONAL MATCH (p)-[:HAS_COMPOUND]->(c:Compound)
        RETURN p.commonName AS nama, p.latinName AS nama_latin,
               p.simplisiaName AS nama_simplisia,
               f.name AS famili,
               collect(DISTINCT c.name) AS senyawa_aktif
        LIMIT $limit
        """
    elif p_enum == Persona.PENELITI:
        cypher = """
        MATCH (p:Herb)
        WHERE any(w IN $words WHERE toLower(p.commonName) CONTAINS w OR toLower(p.latinName) CONTAINS w)
           OR toLower(p.commonName) CONTAINS toLower($query)
           OR toLower(p.latinName) CONTAINS toLower($query)
        OPTIONAL MATCH (p)-[:HAS_COMPOUND]->(c:Compound)
        OPTIONAL MATCH (p)-[:HAS_COMPOUND_CLASS]->(cc:CompoundClass)
        OPTIONAL MATCH (p)-[:HAS_PROTEIN_TARGET]->(pt:ProteinTarget)
        RETURN p.commonName AS nama, p.latinName AS nama_latin,
               p.simplisiaName AS simplisia,
               collect(DISTINCT c.name) AS senyawa_aktif,
               collect(DISTINCT cc.name) AS kelas_senyawa,
               collect(DISTINCT pt.name) AS target_protein
        LIMIT $limit
        """
    else: # Tenaga Medis
        cypher = """
        MATCH (p:Herb)
        WHERE any(w IN $words WHERE toLower(p.commonName) CONTAINS w OR toLower(p.latinName) CONTAINS w)
           OR toLower(p.commonName) CONTAINS toLower($query)
           OR toLower(p.latinName) CONTAINS toLower($query)
        OPTIONAL MATCH (p)-[:HAS_TOXICITY]->(tox:ToxicityCategory)
        OPTIONAL MATCH (p)-[:USED_FOR]->(t:TherapeuticUse)
        RETURN p.commonName AS nama, p.latinName AS nama_latin,
               p.simplisiaName AS simplisia,
               p.macroscopicDesc AS deskripsi_makros,
               p.microscopicDesc AS deskripsi_mikros,
               collect(DISTINCT t.name) AS indikasi_klinis,
               collect(DISTINCT tox.name) AS toksisitas_dan_efek_samping
        LIMIT $limit
        """

    graph_results = _graph_search(query, cypher, words=cleaned_words, limit=graph_limit)
    graph_text = _format_records_to_text(
        graph_results, "Relasi Graph Ensiklopedia"
    )

    return f"{vector_text}\n\n{graph_text}"

# ═══════════════════════════════════════════
# INTENT: EDUKASI (Education Corpus Retrieval)
# ═══════════════════════════════════════════
def retrieve_education_corpus(query: str, limit: int = 5, graph_limit: int = 4, persona: str = "umum") -> str:
    """Hybrid search untuk intent 'edukasi'."""
    # ── STEP 1: Vector Search ──
    vector_results = _vector_search(
        query, "education_materials", "match_education", limit
    )
    vector_text = _format_records_to_text(
        vector_results, "Pencarian Materi Edukasi"
    )

    # ── STEP 2: Graph Search ──
    tag_match = re.search(r'\[target:\s*([^\]]+)\]', query.lower() if query else "")
    if tag_match:
        exact_compound = tag_match.group(1).strip()
        cleaned_words = [exact_compound]
    else:
        cleaned_words = list(set(re.findall(r'\b\w{4,}\b', query.lower() if query else "")))
        if not cleaned_words:
            cleaned_words = [query.lower()] if query else [""]

    p_enum = _normalize_persona(persona)

    # Dual-Path query optimized for schema properties and persona profiles
    if p_enum == Persona.UMUM:
        cypher = """
        OPTIONAL MATCH (h:Herb)
        WHERE any(w IN $words WHERE toLower(h.commonName) CONTAINS w OR w CONTAINS toLower(h.commonName))
        OPTIONAL MATCH (c:Compound)
        WHERE any(w IN $words WHERE toLower(c.name) CONTAINS w OR w CONTAINS toLower(c.name))
        OPTIONAL MATCH (h2:Herb)-[:HAS_COMPOUND]->(c)
        WITH collect(DISTINCT h) + collect(DISTINCT h2) AS merged_herbs
        UNWIND merged_herbs AS final_herb
        WITH DISTINCT final_herb WHERE final_herb IS NOT NULL
        OPTIONAL MATCH (final_herb)-[:USED_FOR]->(t:TherapeuticUse)
        RETURN final_herb.commonName AS topik,
               final_herb.macroscopicDesc AS deskripsi,
               collect(DISTINCT t.name) AS manfaat
        LIMIT $limit
        """
    elif p_enum == Persona.PELAJAR:
        cypher = """
        OPTIONAL MATCH (h:Herb)
        WHERE any(w IN $words WHERE toLower(h.commonName) CONTAINS w OR w CONTAINS toLower(h.commonName))
        OPTIONAL MATCH (c:Compound)
        WHERE any(w IN $words WHERE toLower(c.name) CONTAINS w OR w CONTAINS toLower(c.name))
        OPTIONAL MATCH (h2:Herb)-[:HAS_COMPOUND]->(c)
        WITH collect(DISTINCT h) + collect(DISTINCT h2) AS merged_herbs
        UNWIND merged_herbs AS final_herb
        WITH DISTINCT final_herb WHERE final_herb IS NOT NULL
        OPTIONAL MATCH (final_herb)-[:HAS_COMPOUND]->(comp:Compound)
        RETURN final_herb.commonName AS topik,
               final_herb.simplisiaName AS simplisia,
               collect(DISTINCT comp.name) AS konsep_kunci
        LIMIT $limit
        """
    elif p_enum == Persona.PENELITI:
        cypher = """
        OPTIONAL MATCH (h:Herb)
        WHERE any(w IN $words WHERE toLower(h.commonName) CONTAINS w OR w CONTAINS toLower(h.commonName))
        OPTIONAL MATCH (c:Compound)
        WHERE any(w IN $words WHERE toLower(c.name) CONTAINS w OR w CONTAINS toLower(c.name))
        OPTIONAL MATCH (h2:Herb)-[:HAS_COMPOUND]->(c)
        WITH collect(DISTINCT h) + collect(DISTINCT h2) AS merged_herbs
        UNWIND merged_herbs AS final_herb
        WITH DISTINCT final_herb WHERE final_herb IS NOT NULL
        OPTIONAL MATCH (final_herb)-[:HAS_COMPOUND]->(comp:Compound)
        OPTIONAL MATCH (final_herb)-[:HAS_PROTEIN_TARGET]->(pt:ProteinTarget)
        RETURN final_herb.commonName AS topik,
               collect(DISTINCT comp.name) AS marker_compounds,
               collect(DISTINCT pt.name) AS target_protein
        LIMIT $limit
        """
    else: # Tenaga Medis
        cypher = """
        OPTIONAL MATCH (h:Herb)
        WHERE any(w IN $words WHERE toLower(h.commonName) CONTAINS w OR w CONTAINS toLower(h.commonName))
        OPTIONAL MATCH (c:Compound)
        WHERE any(w IN $words WHERE toLower(c.name) CONTAINS w OR w CONTAINS toLower(c.name))
        OPTIONAL MATCH (h2:Herb)-[:HAS_COMPOUND]->(c)
        WITH collect(DISTINCT h) + collect(DISTINCT h2) AS merged_herbs
        UNWIND merged_herbs AS final_herb
        WITH DISTINCT final_herb WHERE final_herb IS NOT NULL
        OPTIONAL MATCH (final_herb)-[:HAS_TOXICITY]->(tox:ToxicityCategory)
        OPTIONAL MATCH (final_herb)-[:USED_FOR]->(t:TherapeuticUse)
        RETURN final_herb.commonName AS topik,
               collect(DISTINCT t.name) AS indikasi,
               collect(DISTINCT tox.name) AS safety
        LIMIT $limit
        """

    graph_results = _graph_search(query, cypher, words=cleaned_words, limit=graph_limit)

    # Fallback to loose matching if dual-path empty
    if not graph_results:
        logger.info("[Retriever Fallback] Dual-path graph search returned 0 rows. Retrying with loose text matching.")
        if p_enum == Persona.UMUM:
            fallback_cypher = """
            MATCH (h:Herb)
            WHERE toLower(h.commonName) CONTAINS toLower($query) OR toLower($query) CONTAINS toLower(h.commonName)
            OPTIONAL MATCH (h)-[:USED_FOR]->(t:TherapeuticUse)
            RETURN h.commonName AS topik, h.macroscopicDesc AS deskripsi, collect(DISTINCT t.name) AS manfaat
            LIMIT $limit
            """
        elif p_enum == Persona.PELAJAR:
            fallback_cypher = """
            MATCH (h:Herb)
            WHERE toLower(h.commonName) CONTAINS toLower($query) OR toLower($query) CONTAINS toLower(h.commonName)
            OPTIONAL MATCH (h)-[:HAS_COMPOUND]->(c:Compound)
            RETURN h.commonName AS topik, h.simplisiaName AS simplisia, collect(DISTINCT c.name) AS konsep_kunci
            LIMIT $limit
            """
        elif p_enum == Persona.PENELITI:
            fallback_cypher = """
            MATCH (h:Herb)
            WHERE toLower(h.commonName) CONTAINS toLower($query) OR toLower($query) CONTAINS toLower(h.commonName)
            OPTIONAL MATCH (h)-[:HAS_COMPOUND]->(c:Compound)
            OPTIONAL MATCH (h)-[:HAS_PROTEIN_TARGET]->(pt:ProteinTarget)
            RETURN h.commonName AS topik, collect(DISTINCT c.name) AS marker_compounds, collect(DISTINCT pt.name) AS target_protein
            LIMIT $limit
            """
        else:
            fallback_cypher = """
            MATCH (h:Herb)
            WHERE toLower(h.commonName) CONTAINS toLower($query) OR toLower($query) CONTAINS toLower(h.commonName)
            OPTIONAL MATCH (h)-[:HAS_TOXICITY]->(tox:ToxicityCategory)
            OPTIONAL MATCH (h)-[:USED_FOR]->(t:TherapeuticUse)
            RETURN h.commonName AS topik, collect(DISTINCT t.name) AS indikasi, collect(DISTINCT tox.name) AS safety
            LIMIT $limit
            """
        graph_results = _graph_search(query, fallback_cypher, limit=graph_limit)

    graph_text = _format_records_to_text(
        graph_results, "Relasi Graph Topik Edukasi"
    )

    return f"{vector_text}\n\n{graph_text}"
