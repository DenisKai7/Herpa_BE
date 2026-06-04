"""
GraphRAG Retriever - Hybrid search menggunakan Supabase pgvector + Neo4j Cypher.

Menggabungkan dua paradigma pencarian:
1. Cosine Similarity (Semantic Search): Menggunakan pgvector di Supabase
   untuk menemukan dokumen yang secara semantik mirip dengan query.
2. Graph Traversal (Relational Search): Menggunakan Cypher di Neo4j
   untuk menemukan relasi antar entitas (compounds, symptoms, drugs).

Setiap intent memiliki retriever khusus:
- konsultasi: tanaman obat berdasarkan gejala + relasi compound/drug.
- ensiklopedia: informasi detail tanaman + taksonomi/botani.
- edukasi: materi edukasi + relasi antar konsep dengan proteksi multi-label.
"""

import logging
import re
from typing import Any, Optional

from app.core.database import neo4j_driver, supabase
from app.core.embedding import embed_text

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════
# HELPER: VECTOR SEARCH (Supabase pgvector)
# ═══════════════════════════════════════════

def _vector_search(
    query: str,
    table: str,
    match_function: str,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """
    Melakukan pencarian semantik menggunakan Supabase RPC (pgvector cosine similarity).

    Supabase harus memiliki SQL function (match_function) yang menerima:
    - query_embedding: vector(768)
    - match_count: int
    - match_threshold: float
    """
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
    """
    Menjalankan Cypher query di Neo4j untuk mendapatkan relasi antar entitas.
    Menggunakan driver.execute_query() dengan built-in auto-retry untuk koneksi defunct.
    """
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
        logger.error(
            f"Graph database connectivity lost or query failed: {e}",
            exc_info=True,
        )
        return []


def _format_records_to_text(
    records: list[dict[str, Any]],
    source_label: str = "Database",
) -> str:
    """
    Mengubah list of dict menjadi teks terstruktur untuk dikirim ke LLM.
    """
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

def content_based_recommendation(query: str, limit: int = 5) -> str:
    """
    Hybrid search untuk intent 'konsultasi' dengan pencarian gejala dan multi-label.
    """
    # ── STEP 1: Vector Search ──
    vector_results = _vector_search(query, "plants", "match_plants", limit)
    vector_text = _format_records_to_text(
        vector_results, "Pencarian Semantik Tanaman Obat"
    )

    # ── STEP 2: Graph Search ──
    # Strict Tag Isolation Layer
    tag_match = re.search(r'\[target:\s*([^\]]+)\]', query.lower() if query else "")
    if tag_match:
        # Clean any potential spaces or underscores
        exact_compound = tag_match.group(1).strip()
        cleaned_words = [exact_compound]
        logger.info(f"[Retriever Strict Match] Isolated visual target compound: {cleaned_words}")
    else:
        # Standard text chat fallback tokenization
        cleaned_words = list(set(re.findall(r'\b\w{4,}\b', query.lower() if query else "")))
        if not cleaned_words:
            cleaned_words = [query.lower()] if query else [""]

    cypher = """
    MATCH (p)-[:TREATS]->(s:Symptom)
    WHERE (p:Plant OR p:Herb)
      AND (any(w IN $words WHERE toLower(s.name) CONTAINS w OR w CONTAINS toLower(s.name))
       OR any(w IN $words WHERE toLower(p.name) CONTAINS w OR w CONTAINS toLower(p.name))
       OR toLower(s.name) CONTAINS toLower($query)
       OR toLower(p.name) CONTAINS toLower($query))
    OPTIONAL MATCH (p)-[:HAS_COMPOUND]->(c:Compound)
    OPTIONAL MATCH (c)-[:INTERACTS_WITH]->(d:Drug)
    RETURN p.name AS tanaman, p.nama_latin AS nama_latin,
           collect(DISTINCT s.name) AS gejala_terkait,
           collect(DISTINCT c.name) AS senyawa_aktif,
           collect(DISTINCT d.name) AS interaksi_obat
    LIMIT 5
    """
    graph_results = _graph_search(query, cypher, words=cleaned_words)
    graph_text = _format_records_to_text(
        graph_results, "Relasi Graph Tanaman-Gejala-Senyawa"
    )

    return f"{vector_text}\n\n{graph_text}"


# ═══════════════════════════════════════════
# INTENT: ENSIKLOPEDIA (Encyclopedia Search)
# ═══════════════════════════════════════════

def search_encyclopedia(query: str, limit: int = 5) -> str:
    """
    Hybrid search untuk intent 'ensiklopedia' dengan proteksi multi-label.
    """
    # ── STEP 1: Vector Search ──
    vector_results = _vector_search(
        query, "encyclopedia", "match_encyclopedia", limit
    )
    vector_text = _format_records_to_text(vector_results, "Pencarian Ensiklopedia")

    # ── STEP 2: Graph Search ──
    # Strict Tag Isolation Layer
    tag_match = re.search(r'\[target:\s*([^\]]+)\]', query.lower() if query else "")
    if tag_match:
        # Clean any potential spaces or underscores
        exact_compound = tag_match.group(1).strip()
        cleaned_words = [exact_compound]
        logger.info(f"[Retriever Strict Match] Isolated visual target compound: {cleaned_words}")
    else:
        # Standard text chat fallback tokenization
        cleaned_words = list(set(re.findall(r'\b\w{4,}\b', query.lower() if query else "")))
        if not cleaned_words:
            cleaned_words = [query.lower()] if query else [""]

    cypher = """
    MATCH (p)
    WHERE (p:Plant OR p:Herb)
      AND (any(w IN $words WHERE toLower(p.name) CONTAINS w OR w CONTAINS toLower(p.name))
       OR any(w IN $words WHERE toLower(p.nama_latin) CONTAINS w OR w CONTAINS toLower(p.nama_latin))
       OR toLower(p.name) CONTAINS toLower($query)
       OR toLower(p.nama_latin) CONTAINS toLower($query))
    OPTIONAL MATCH (p)-[:BELONGS_TO]->(f:Family)
    OPTIONAL MATCH (p)-[:HAS_COMPOUND]->(c:Compound)
    OPTIONAL MATCH (p)-[:FOUND_IN]->(r:Region)
    RETURN p.name AS nama, p.nama_latin AS nama_latin,
           p.description AS deskripsi,
           f.name AS famili,
           collect(DISTINCT c.name) AS senyawa_aktif,
           collect(DISTINCT r.name) AS daerah_asal
    LIMIT 5
    """
    graph_results = _graph_search(query, cypher, words=cleaned_words)
    graph_text = _format_records_to_text(
        graph_results, "Relasi Graph Ensiklopedia"
    )

    return f"{vector_text}\n\n{graph_text}"


# ═══════════════════════════════════════════
# INTENT: EDUKASI (Education Corpus Retrieval)
# ═══════════════════════════════════════════

def retrieve_education_corpus(query: str, limit: int = 5) -> str:
    """
    Hybrid search untuk intent 'edukasi'.
    Mencari secara independen dari sisi Tanaman (Herb) maupun Senyawa (Compound)
    untuk menghindari bottleneck pencarian multimodal RAG.
    """
    # ── STEP 1: Vector Search (Supabase pgvector) ──
    vector_results = _vector_search(
        query, "education_materials", "match_education", limit
    )
    vector_text = _format_records_to_text(
        vector_results, "Pencarian Materi Edukasi"
    )

    # ── STEP 2: Graph Search (Neo4j Independent Dual-Path Match) ──
    # Strict Tag Isolation Layer
    tag_match = re.search(r'\[target:\s*([^\]]+)\]', query.lower() if query else "")
    if tag_match:
        # Clean any potential spaces or underscores
        exact_compound = tag_match.group(1).strip()
        cleaned_words = [exact_compound]
        logger.info(f"[Retriever Strict Match] Isolated visual target compound: {cleaned_words}")
    else:
        # Standard text chat fallback tokenization
        cleaned_words = list(set(re.findall(r'\b\w{4,}\b', query.lower() if query else "")))
        if not cleaned_words:
            cleaned_words = [query.lower()] if query else [""]

    # Kueri Utama: Jalur pencarian mandiri (Herb dan Compound dipisah)
    cypher = """
    // Jalur 1: Cari Herb langsung jika kata kunci cocok dengan nama tanaman
    OPTIONAL MATCH (h:Herb)
    WHERE any(w IN $words WHERE toLower(h.name) CONTAINS w OR w CONTAINS toLower(h.name))

    // Jalur 2: Cari Compound jika kata kunci cocok dengan nama senyawa gambar (e.g. Curcumin)
    OPTIONAL MATCH (c:Compound)
    WHERE any(w IN $words WHERE toLower(c.name) CONTAINS w OR w CONTAINS toLower(c.name))
    // Hubungkan senyawa yang cocok ke node Herb induknya
    OPTIONAL MATCH (h2:Herb)-[:HAS_COMPOUND]->(c)

    // Gabungkan hasil tanaman dari Jalur 1 dan Jalur 2 secara aman
    WITH collect(DISTINCT h) + collect(DISTINCT h2) AS merged_herbs
    UNWIND merged_herbs AS final_herb
    WITH DISTINCT final_herb WHERE final_herb IS NOT NULL

    // Ambil seluruh daftar senyawa aktif dari tanaman yang terpilih untuk konteks LLM
    OPTIONAL MATCH (final_herb)-[:HAS_COMPOUND]->(comp:Compound)
    RETURN final_herb.name AS topik, final_herb.name AS deskripsi,
           collect(DISTINCT comp.name) AS konsep_kunci
    LIMIT 5
    """
    graph_results = _graph_search(query, cypher, words=cleaned_words)

    # Fallback Kueri: Jika jalur independen kosong, gunakan pencarian string pencocokan jarak longgar
    if not graph_results:
        logger.info("[Retriever Fallback] Dual-path graph search returned 0 rows. Retrying with loose text matching.")
        fallback_cypher = """
        MATCH (h:Herb)
        WHERE toLower(h.name) CONTAINS toLower($query) OR toLower($query) CONTAINS toLower(h.name)
        OPTIONAL MATCH (h)-[:HAS_COMPOUND]->(c:Compound)
        RETURN h.name AS topik, h.name AS deskripsi,
               collect(DISTINCT c.name) AS konsep_kunci
        LIMIT 5
        """
        graph_results = _graph_search(query, fallback_cypher)
        
    graph_text = _format_records_to_text(
        graph_results, "Relasi Graph Topik Edukasi"
    )

    return f"{vector_text}\n\n{graph_text}"