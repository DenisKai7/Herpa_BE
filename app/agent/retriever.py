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
- edukasi: materi edukasi + relasi antar konsep.
"""

import logging
from typing import Any

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

    Args:
        query: Teks query pengguna.
        table: Nama tabel target (untuk logging).
        match_function: Nama RPC function di Supabase (e.g., 'match_plants').
        limit: Jumlah hasil maksimal.

    Returns:
        List of dict: Hasil pencarian yang relevan.
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

def _graph_search(query: str, cypher_query: str) -> list[dict[str, Any]]:
    """
    Menjalankan Cypher query di Neo4j untuk mendapatkan relasi antar entitas.

    Menggunakan driver.execute_query() yang memiliki built-in auto-retry
    untuk koneksi defunct/expired (SessionExpired, ServiceUnavailable).

    Graph model yang diharapkan:
    (:Herb)-[:HAS_COMPOUND]->(:Compound)
    (:Herb)-[:USED_FOR]->(:TherapeuticUse)
    (:Compound)-[:INTERACTS_WITH]->(:Drug)

    Args:
        query: Parameter $query untuk Cypher query.
        cypher_query: String Cypher query yang akan dijalankan.

    Returns:
        List of dict: Record hasil Cypher query. Empty list on failure.
    """
    try:
        records, _, _ = neo4j_driver.execute_query(
            cypher_query,
            parameters_={"query": query},
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

    Memfilter field yang tidak relevan (embedding, id) dan memformat
    setiap record menjadi key-value pairs yang mudah dibaca.

    Args:
        records: List of dict dari hasil pencarian.
        source_label: Label sumber untuk header teks.

    Returns:
        String terformat untuk konteks LLM.
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
# Mencari tanaman/herbal berdasarkan gejala/keluhan
# ═══════════════════════════════════════════

def content_based_recommendation(query: str, limit: int = 5) -> str:
    """
    Hybrid search untuk intent 'konsultasi'.

    Pipeline:
    1. Vector search: Cari tanaman yang secara semantik relevan dengan gejala.
    2. Graph search: Temukan relasi tambahan (compound, interaksi obat).

    Args:
        query: Deskripsi gejala/keluhan dari pengguna.
        limit: Jumlah hasil maksimal dari vector search.

    Returns:
        String konteks gabungan dari vector + graph search.
    """
    # ── STEP 1: Vector Search (Supabase pgvector) ──
    vector_results = _vector_search(query, "plants", "match_plants", limit)
    vector_text = _format_records_to_text(
        vector_results, "Pencarian Semantik Tanaman Obat"
    )

    # ── STEP 2: Graph Search (Neo4j) ──
    cypher = """
    MATCH (p:Plant)-[:TREATS]->(s:Symptom)
    WHERE toLower(s.name) CONTAINS toLower($query)
       OR toLower(p.name) CONTAINS toLower($query)
    OPTIONAL MATCH (p)-[:HAS_COMPOUND]->(c:Compound)
    OPTIONAL MATCH (c)-[:INTERACTS_WITH]->(d:Drug)
    RETURN p.name AS tanaman, p.nama_latin AS nama_latin,
           collect(DISTINCT s.name) AS gejala_terkait,
           collect(DISTINCT c.name) AS senyawa_aktif,
           collect(DISTINCT d.name) AS interaksi_obat
    LIMIT 5
    """
    graph_results = _graph_search(query, cypher)
    graph_text = _format_records_to_text(
        graph_results, "Relasi Graph Tanaman-Gejala-Senyawa"
    )

    return f"{vector_text}\n\n{graph_text}"


# ═══════════════════════════════════════════
# INTENT: ENSIKLOPEDIA (Encyclopedia Search)
# Mencari informasi detail tentang tanaman/senyawa tertentu
# ═══════════════════════════════════════════

def search_encyclopedia(query: str, limit: int = 5) -> str:
    """
    Hybrid search untuk intent 'ensiklopedia'.

    Pipeline:
    1. Vector search: Cari entri ensiklopedia yang paling relevan.
    2. Graph search: Temukan klasifikasi taksonomi dan relasi botani.

    Args:
        query: Kata kunci pencarian ensiklopedia.
        limit: Jumlah hasil maksimal dari vector search.

    Returns:
        String konteks gabungan dari vector + graph search.
    """
    # ── STEP 1: Vector Search ──
    vector_results = _vector_search(
        query, "encyclopedia", "match_encyclopedia", limit
    )
    vector_text = _format_records_to_text(vector_results, "Pencarian Ensiklopedia")

    # ── STEP 2: Graph Search ──
    cypher = """
    MATCH (p:Plant)
    WHERE toLower(p.name) CONTAINS toLower($query)
       OR toLower(p.nama_latin) CONTAINS toLower($query)
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
    graph_results = _graph_search(query, cypher)
    graph_text = _format_records_to_text(
        graph_results, "Relasi Graph Ensiklopedia"
    )

    return f"{vector_text}\n\n{graph_text}"


# ═══════════════════════════════════════════
# INTENT: EDUKASI (Education Corpus Retrieval)
# Mencari materi edukasi kimia/farmasi/biologi
# ═══════════════════════════════════════════

def retrieve_education_corpus(query: str, limit: int = 5) -> str:
    """
    Hybrid search untuk intent 'edukasi'.

    Pipeline:
    1. Vector search: Cari materi edukasi yang paling relevan.
    2. Graph search: Cari relasi konsep kimia/biologi.

    Args:
        query: Topik atau pertanyaan edukasi.
        limit: Jumlah hasil maksimal dari vector search.

    Returns:
        String konteks gabungan dari vector + graph search.
    """
    # ── STEP 1: Vector Search ──
    vector_results = _vector_search(
        query, "education_materials", "match_education", limit
    )
    vector_text = _format_records_to_text(
        vector_results, "Pencarian Materi Edukasi"
    )

    # ── STEP 2: Graph Search ──
    cypher = """
    MATCH (h:Herb)
    WHERE toLower(h.name) CONTAINS toLower($query)
    OPTIONAL MATCH (h)-[:HAS_COMPOUND]->(c:Compound)
    OPTIONAL MATCH (h)-[:USED_FOR]->(t:TherapeuticUse)
    RETURN h.name AS topik, h.name AS deskripsi,
           collect(DISTINCT c.name) AS konsep_kunci,
           collect(DISTINCT t.name) AS topik_terkait
    LIMIT 5
    """
    graph_results = _graph_search(query, cypher)
    graph_text = _format_records_to_text(
        graph_results, "Relasi Graph Topik Edukasi"
    )

    return f"{vector_text}\n\n{graph_text}"
