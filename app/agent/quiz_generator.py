"""
Quiz Generator - Agentic Tool-Calling untuk pembuatan kuis interaktif.

Pipeline yang robust dan fault-tolerant:
1. NLP Preprocessing: Membersihkan noise percakapan dari prompt pengguna,
   mengekstrak topik inti dan jumlah soal secara dinamis.
2. Adaptive Hybrid Retrieval: Mengadaptasi strategi pencarian berdasarkan
   cakupan query (spesifik vs. umum/broad).
3. Scope-Aware System Prompt: Menyesuaikan instruksi LLM berdasarkan
   apakah topik spesifik atau umum.
4. Multi-Layer Parsing + Synthetic Fallback: tool_calls -> regex JSON
   extraction -> local synthetic quiz generator.

Temperature 0.2 untuk variasi soal yang terkontrol.
"""

import json
import logging
import re
from typing import Any, Optional

from huggingface_hub import InferenceClient

from app.core.config import settings
from app.core.database import neo4j_driver, supabase
from app.core.embedding import embed_text
from app.models.quiz_schemas import QuizResponse

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════
# LLM CLIENT (Shared, Singleton) - HuggingFace Inference API
# ═══════════════════════════════════════════
_client = InferenceClient(
    provider="auto",
    api_key=settings.HF_API_TOKEN,
)


# ═══════════════════════════════════════════
# [1] NLP PREPROCESSING — Keyword & Intent Cleaning
# ═══════════════════════════════════════════

# Noise words/phrases commonly found in quiz generation prompts
_NOISE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"\b(?:tolong|coba|mohon|bisa|bisakah|dong|ya|yuk|ayo|silakan|minta"
        r"|bantu|bantuin|bikinin|carikan|buatkan|buat|bikin|generate|beri|berikan"
        r"|tampilkan|tunjukkan|kasih)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:kuis|quiz|soal|pertanyaan|latihan|ujian)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:tentang|mengenai|terkait|seputar|perihal|berkaitan dengan"
        r"|yang berkaitan|yang berhubungan|dengan topik|dengan materi)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:tingkat kesulitan|kesulitan|difficulty)\s*(?:nya)?\s*"
        r"(?:mudah|menengah|sedang|sulit|tinggi|rendah|hots|easy|medium|hard)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:dengan|yang|untuk|dari|ke|di|pada)\b",
        re.IGNORECASE,
    ),
]

# Pattern to extract requested number of questions
_JUMLAH_PATTERN = re.compile(
    r"(\d+)\s*(?:buah|butir|nomor|nomer)?\s*(?:soal|pertanyaan|kuis|quiz|question)",
    re.IGNORECASE,
)
_JUMLAH_PATTERN_ALT = re.compile(
    r"(?:soal|pertanyaan|kuis|quiz|question)\s*(?:sebanyak)?\s*(\d+)",
    re.IGNORECASE,
)


def _extract_jumlah_soal(raw_prompt: str, default: int = 3) -> int:
    """
    Mengekstrak jumlah soal yang diminta dari prompt pengguna.

    Mencari pola seperti "5 soal", "soal 10", "3 pertanyaan", dll.
    Clamp ke range [1, 10] untuk keamanan.

    Args:
        raw_prompt: Prompt asli dari pengguna (belum dibersihkan).
        default: Jumlah default jika tidak terdeteksi.

    Returns:
        Integer jumlah soal (1-10).
    """
    match = _JUMLAH_PATTERN.search(raw_prompt)
    if not match:
        match = _JUMLAH_PATTERN_ALT.search(raw_prompt)
    if match:
        count = int(match.group(1))
        clamped = max(1, min(count, 10))
        logger.info(
            f"Extracted jumlah_soal={clamped} from prompt "
            f"(raw={count}, clamped={clamped})."
        )
        return clamped
    return default


def _clean_topic(raw_prompt: str) -> str:
    """
    Membersihkan noise percakapan dari prompt dan mengekstrak topik inti.

    Contoh:
    - "tolong buatkan kuis 5 soal mengenai tanaman herbal" -> "tanaman herbal"
    - "kuis tentang kurkumin pada temulawak" -> "kurkumin temulawak"
    - "buat soal kimia organik tingkat kesulitan tinggi" -> "kimia organik"

    Args:
        raw_prompt: Prompt asli dari pengguna.

    Returns:
        String topik yang sudah dibersihkan.
    """
    cleaned = raw_prompt.strip()

    # Remove number-of-questions phrases before general cleaning
    cleaned = _JUMLAH_PATTERN.sub("", cleaned)
    cleaned = _JUMLAH_PATTERN_ALT.sub("", cleaned)

    # Apply noise pattern removal
    for pattern in _NOISE_PATTERNS:
        cleaned = pattern.sub(" ", cleaned)

    # Collapse whitespace and strip
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    # If cleaning removed everything, return the original prompt
    if not cleaned or len(cleaned) < 2:
        logger.warning(
            f"Topic cleaning resulted in empty string, "
            f"using original prompt: '{raw_prompt[:60]}'"
        )
        return raw_prompt.strip()

    logger.info(f"Topic cleaned: '{raw_prompt[:60]}' -> '{cleaned}'")
    return cleaned


# ═══════════════════════════════════════════
# [2] ADAPTIVE HYBRID RETRIEVAL
# ═══════════════════════════════════════════

# Pre-defined general corpus summaries for broad domains
_GENERAL_CORPUS: dict[str, str] = {
    "kimia": (
        "Kimia farmasi mencakup studi tentang senyawa kimia yang terdapat dalam "
        "tanaman obat. Cabang utama meliputi: Fitokimia (studi metabolit sekunder "
        "seperti alkaloid, flavonoid, terpenoid, saponin, dan tanin), Farmakognosi "
        "(identifikasi dan standarisasi bahan alam), dan Kimia Medisinal "
        "(hubungan struktur-aktivitas/SAR senyawa bioaktif). Metabolit sekunder "
        "utama: alkaloid (analgesik, antimalaria), flavonoid (antioksidan, "
        "antiinflamasi), terpenoid (antikanker, antimikroba), saponin "
        "(immunomodulator), dan tanin (astringen, antidiare). Teknik analisis "
        "penting: HPLC, GC-MS, UV-Vis spektrofotometri, dan uji aktivitas "
        "biologis (IC50, MIC, LD50)."
    ),
    "tanaman_obat": (
        "Indonesia memiliki lebih dari 30.000 spesies tumbuhan, sekitar 7.000 "
        "di antaranya digunakan sebagai obat tradisional. Tanaman obat utama "
        "meliputi: Kunyit (Curcuma longa — kurkumin, antiinflamasi, hepatoprotektor), "
        "Temulawak (Curcuma xanthorrhiza — xanthorrhizol, hepatoprotektor), "
        "Jahe (Zingiber officinale — gingerol, antiemetik, analgesik), "
        "Sambiloto (Andrographis paniculata — andrografolid, immunomodulator), "
        "Mengkudu (Morinda citrifolia — skopoletin, antihipertensi), "
        "Pegagan (Centella asiatica — asiatikosida, penyembuhan luka), dan "
        "Kumis Kucing (Orthosiphon stamineus — sinensetin, diuretik). "
        "Bagian tumbuhan yang digunakan: rhizoma, folium, radix, cortex, flos, fructus, semen."
    ),
    "farmasi": (
        "Farmasi herbal mencakup: Farmakodinamik (mekanisme aksi senyawa pada "
        "reseptor dan enzim), Farmakokinetik (ADME — absorpsi, distribusi, "
        "metabolisme, ekskresi), Interaksi Obat-Herbal (induksi/inhibisi enzim "
        "sitokrom P450, sinergisme dan antagonisme), Formulasi Sediaan "
        "(simplisia, ekstrak, tinktur, kapsul, tablet), serta Standardisasi "
        "dan Quality Control (kadar air, kadar abu, kandungan senyawa marker). "
        "Kontraindikasi umum: kehamilan, menyusui, gangguan hepar/renal, "
        "pediatri, dan interaksi dengan antikoagulan."
    ),
    "herbal": (
        "Obat herbal Indonesia diklasifikasikan dalam 3 kategori BPOM: "
        "Jamu (berdasarkan pengalaman empiris turun-temurun), Obat Herbal "
        "Terstandar/OHT (telah melalui uji praklinis), dan Fitofarmaka "
        "(telah melalui uji klinis). Sediaan herbal meliputi: decocta "
        "(rebusan), infusa (seduhan), tinktur (ekstrak alkohol), dan "
        "maserasinya. Pelarut ekstraksi umum: etanol, metanol, air, "
        "etil asetat, dan n-heksana. Uji bioaktivitas standar: DPPH "
        "(antioksidan), difusi cakram (antimikroba), MTT assay "
        "(sitotoksisitas), dan uji toleransi glukosa (antidiabetes)."
    ),
}

# Keywords that map to general corpus categories
_BROAD_KEYWORD_MAP: dict[str, list[str]] = {
    "kimia": [
        "kimia", "chemistry", "senyawa", "reaksi", "molekul",
        "organik", "anorganik", "fitokimia", "metabolit",
    ],
    "tanaman_obat": [
        "tanaman obat", "tanaman herbal", "tumbuhan obat", "herba",
        "simplisia", "jamu", "rempah", "medicinal plant",
    ],
    "farmasi": [
        "farmasi", "farmakologi", "farmakokinetik", "farmakodinamik",
        "obat", "dosis", "pharmaceutical", "apoteker",
    ],
    "herbal": [
        "herbal", "obat herbal", "obat tradisional", "pengobatan tradisional",
        "ramuan", "ekstrak", "decocta", "infusa",
    ],
}


def _detect_broad_domain(topic: str) -> Optional[str]:
    """
    Mendeteksi apakah topik termasuk kategori umum/broad.

    Args:
        topic: Topik yang sudah dibersihkan.

    Returns:
        Key domain dari _GENERAL_CORPUS jika broad, None jika spesifik.
    """
    topic_lower = topic.lower()
    for domain, keywords in _BROAD_KEYWORD_MAP.items():
        for kw in keywords:
            if kw in topic_lower:
                return domain
    return None


def _vector_search_quiz(
    query: str,
    limit: int = 5,
    threshold: float = 0.7,
) -> list[dict[str, Any]]:
    """
    Pencarian semantik via Supabase pgvector untuk konteks kuis.

    Args:
        query: Teks query yang sudah dibersihkan.
        limit: Jumlah hasil maksimal.
        threshold: Similarity threshold minimum.

    Returns:
        List of dict hasil pencarian.
    """
    try:
        query_embedding = embed_text(query)

        # Try education materials first
        result = supabase.rpc("match_education", {
            "query_embedding": query_embedding,
            "match_count": limit,
            "match_threshold": threshold,
        }).execute()

        if result.data:
            logger.info(
                f"Quiz vector search (education): "
                f"{len(result.data)} results (threshold={threshold})."
            )
            return result.data

        # Fallback to plants table
        result = supabase.rpc("match_plants", {
            "query_embedding": query_embedding,
            "match_count": limit,
            "match_threshold": threshold,
        }).execute()

        if result.data:
            logger.info(
                f"Quiz vector search (plants): "
                f"{len(result.data)} results (threshold={threshold})."
            )
            return result.data

        return []

    except Exception as e:
        logger.error(f"Quiz vector search error: {e}", exc_info=True)
        return []


def _graph_search_quiz(topic: str) -> list[dict[str, Any]]:
    """
    Pencarian graph Neo4j untuk konteks kuis yang mendalam.

    Mencari relasi Herb->Compound dan Herb->TherapeuticUse
    sesuai dengan schema aktual database Neo4j.

    Args:
        topic: Topik kuis yang sudah dibersihkan.

    Returns:
        List of dict hasil Cypher query.
    """
    cypher = """
    MATCH (h:Herb)
    WHERE toLower(h.name) CONTAINS toLower($query)
    OPTIONAL MATCH (h)-[:HAS_COMPOUND]->(c:Compound)
    OPTIONAL MATCH (h)-[:USED_FOR]->(t:TherapeuticUse)
    RETURN h.name AS topik,
           h.name AS deskripsi,
           collect(DISTINCT c.name) AS konsep_kunci,
           collect(DISTINCT t.name) AS topik_terkait
    LIMIT 5
    """
    try:
        records, _, _ = neo4j_driver.execute_query(
            cypher,
            parameters_={"query": topic},
        )
        result = [record.data() for record in records]
        logger.info(f"Quiz graph search: {len(result)} records for '{topic[:40]}'.")
        return result
    except Exception as e:
        logger.error(
            f"Quiz graph search — Neo4j connectivity lost or query failed: {e}",
            exc_info=True,
        )
        return []


def _broad_graph_search(domain: str) -> list[dict[str, Any]]:
    """
    Pencarian graph wildcard untuk topik umum/broad.

    Mengambil sampel node dari Neo4j tanpa filter spesifik.

    Args:
        domain: Domain yang terdeteksi (kimia/tanaman_obat/farmasi/herbal).

    Returns:
        List of dict hasil Cypher query.
    """
    cypher = """
    MATCH (h:Herb)
    OPTIONAL MATCH (h)-[:HAS_COMPOUND]->(c:Compound)
    OPTIONAL MATCH (h)-[:USED_FOR]->(t:TherapeuticUse)
    RETURN h.name AS topik,
           h.name AS deskripsi,
           collect(DISTINCT c.name) AS konsep_kunci,
           collect(DISTINCT t.name) AS topik_terkait
    LIMIT 8
    """
    try:
        records, _, _ = neo4j_driver.execute_query(cypher)
        result = [record.data() for record in records]
        logger.info(
            f"Broad graph search for domain '{domain}': "
            f"{len(result)} records."
        )
        return result
    except Exception as e:
        logger.error(
            f"Broad graph search — Neo4j connectivity lost or query failed: {e}",
            exc_info=True,
        )
        return []


def _format_records(
    records: list[dict[str, Any]],
    source_label: str,
) -> str:
    """
    Mengubah list of dict menjadi teks terstruktur untuk konteks LLM.

    Args:
        records: List of dict dari hasil pencarian.
        source_label: Label sumber untuk header.

    Returns:
        String terformat.
    """
    if not records:
        return ""

    lines: list[str] = [f"[{source_label} - {len(records)} hasil]:"]
    for i, rec in enumerate(records, 1):
        parts: list[str] = []
        for key, value in rec.items():
            if value is not None and key not in ("embedding", "id", "similarity"):
                parts.append(f"  {key}: {value}")
        lines.append(f"\n--- Hasil #{i} ---")
        lines.extend(parts)
    return "\n".join(lines)


def _retrieve_quiz_context(topic: str) -> tuple[str, bool, bool]:
    """
    Adaptive hybrid retrieval: mengadaptasi strategi berdasarkan cakupan topik.

    Strategy:
    - Case A (Specific): Vector search (threshold 0.7) + Neo4j graph traversal.
    - Case B (Broad/General): Lowered vector search -> broad graph search
      -> general knowledge corpus fallback.

    Graceful degradation: jika Neo4j drop/unreachable, pipeline tetap
    berjalan menggunakan data vector-only dari Supabase.

    Args:
        topic: Topik kuis yang sudah dibersihkan.

    Returns:
        Tuple (context_string, is_broad, graph_available):
        - is_broad=True jika topik umum.
        - graph_available=True jika graph data berhasil diambil.
    """
    # ── STEP 1: Try specific vector search (high threshold) ──
    vector_results = _vector_search_quiz(topic, limit=5, threshold=0.7)
    graph_results = _graph_search_quiz(topic)

    vector_text = _format_records(vector_results, "Pencarian Semantik")
    graph_text = _format_records(graph_results, "Relasi Graph Database")

    graph_available = bool(graph_results)
    has_specific_data = bool(vector_results) or graph_available

    if has_specific_data:
        context = "\n\n".join(filter(None, [vector_text, graph_text]))
        logger.info(
            f"Specific retrieval successful: "
            f"{len(vector_results)} vector + {len(graph_results)} graph results."
        )
        return context, False, graph_available

    # ── STEP 2: Fallback — lowered threshold vector search ──
    logger.info(
        f"No specific results for '{topic[:40]}', "
        "trying lowered threshold (0.4)..."
    )
    vector_results_low = _vector_search_quiz(topic, limit=8, threshold=0.4)
    vector_text_low = _format_records(vector_results_low, "Pencarian Semantik (Diperluas)")

    if vector_results_low:
        broad_graph = _broad_graph_search("general")
        broad_graph_text = _format_records(broad_graph, "Data Umum Database")
        broad_graph_ok = bool(broad_graph)
        context = "\n\n".join(filter(None, [vector_text_low, broad_graph_text]))
        logger.info(
            f"Lowered-threshold retrieval: {len(vector_results_low)} results."
        )
        return context, True, broad_graph_ok

    # ── STEP 3: Fallback — broad graph search + general corpus ──
    logger.info(
        f"Vector search empty for '{topic[:40]}', "
        "falling back to broad graph + general corpus..."
    )
    domain = _detect_broad_domain(topic)
    broad_graph = _broad_graph_search(domain or "general")
    broad_graph_text = _format_records(broad_graph, "Data Umum Database")
    broad_graph_ok = bool(broad_graph)

    # Inject pre-defined general corpus summary
    corpus_text = ""
    if domain and domain in _GENERAL_CORPUS:
        corpus_text = f"[Ringkasan Pengetahuan Domain '{domain.upper()}']:\n{_GENERAL_CORPUS[domain]}"
        logger.info(f"Injected general corpus for domain: '{domain}'.")
    else:
        # Combine all general corpuses as catch-all
        all_corpus = "\n\n".join(
            f"[{k.upper()}]: {v}" for k, v in _GENERAL_CORPUS.items()
        )
        corpus_text = f"[Ringkasan Pengetahuan Umum Farmasi & Tanaman Obat]:\n{all_corpus}"
        logger.info("Injected full general corpus (no specific domain matched).")

    context = "\n\n".join(filter(None, [broad_graph_text, corpus_text]))
    return context, True, broad_graph_ok


# ═══════════════════════════════════════════
# [3] SCOPE-AWARE SYSTEM PROMPT COMPILER
# ═══════════════════════════════════════════

def _build_quiz_system_prompt(
    context_data: str,
    jumlah_soal: int,
    ai_mode: str,
    is_broad: bool,
    topic: str,
    graph_available: bool = True,
    file_context: Optional[str] = None,
) -> str:
    """
    Membangun system prompt untuk quiz generation yang scope-aware.

    Prompt disesuaikan berdasarkan:
    - is_broad: True -> soal konseptual/fundamental. False -> soal case-study mendalam.
    - ai_mode: persona (Pelajar -> bahasa ringan, Tenaga Medis -> terminologi klinis, dll).
    - graph_available: False -> instruksi khusus untuk adaptasi ke teks-only context.

    Args:
        context_data: Konteks dari adaptive retrieval.
        jumlah_soal: Jumlah soal yang diminta.
        ai_mode: Persona AI (Pelajar, Tenaga Medis, Peneliti, Umum).
        is_broad: True jika topik umum/general.
        topic: Topik yang sudah dibersihkan (untuk referensi).
        graph_available: True jika data graph Neo4j berhasil diambil.
        file_context: Teks dari file upload (opsional).

    Returns:
        System prompt string.
    """
    file_instruction = ""
    if file_context:
        # Truncate to 4000 chars to prevent context window overflow
        truncated = file_context[:4000]
        file_instruction = f"""

[DATA KONTEKS BERKAS/FILE YANG DIUNGGAH OLEH USER]
{truncated}

CRITICAL PROTOCOL: User telah melampirkan berkas dokumen/gambar medis. Anda WAJIB memprioritaskan konten file ini saat membuat soal kuis. Buat pertanyaan yang langsung menguji pemahaman konsep, fakta, dan informasi yang terdapat dalam file yang diunggah.

ATURAN PEMBUATAN KUIS BERBASIS FILE:
1. Buat pertanyaan terutama dari konten file yang diunggah.
2. Fokus pada konsep kunci, definisi, mekanisme, dan fakta dari file.
3. Jika file berisi data medis/farmasi spesifik, gunakan sebagai materi soal.
4. Lengkapi dengan pengetahuan database hanya jika konten file tidak cukup.
5. Pastikan semua pertanyaan valid dan dapat dijawab berdasarkan konten file."""

    # Scope-specific generation instructions
    if is_broad:
        scope_instruction = (
            "SCOPE: TOPIK UMUM/GENERAL.\n"
            "- Buat soal KONSEPTUAL dan FUNDAMENTAL yang mencakup prinsip dasar "
            "dari domain yang diminta.\n"
            "- Fokus pada pemahaman definisi, klasifikasi, fungsi umum, dan "
            "perbandingan antar konsep.\n"
            "- Gunakan data dari konteks sebagai bahan soal, tetapi boleh "
            "menyusun soal yang menguji pemahaman lintas-konsep.\n"
            "- Soal harus bervariasi: definisi, perbandingan, sebab-akibat, "
            "dan penerapan.\n"
            f"- Topik utama yang diminta pengguna: \"{topic}\"."
        )
    else:
        scope_instruction = (
            "SCOPE: TOPIK SPESIFIK.\n"
            "- Buat soal MENDALAM dan CASE-STUDY style yang fokus pada mekanisme, "
            "senyawa spesifik, dan interaksi klinis dari data konteks.\n"
            "- Sertakan detail seperti nama senyawa aktif, mekanisme aksi, "
            "efek farmakologis, dan interaksi obat yang ada dalam data.\n"
            "- Soal harus menguji kemampuan analisis dan penerapan, "
            "bukan sekadar hafalan.\n"
            f"- Topik spesifik: \"{topic}\"."
        )

    # Persona language adaptation
    persona_map: dict[str, str] = {
        "Tenaga Medis": (
            "Gunakan terminologi klinis dan farmakologis yang presisi. "
            "Soal harus setara level kompetensi tenaga kesehatan profesional."
        ),
        "Peneliti": (
            "Gunakan bahasa ilmiah formal. Sertakan nama latin, kelas senyawa, "
            "dan terminologi metodologis (IC50, GC-MS, HPLC) dalam soal."
        ),
        "Pelajar": (
            "Gunakan bahasa yang edukatif dan mudah dipahami. "
            "Sertakan pembahasan yang menjelaskan konsep step-by-step. "
            "Cocok untuk mahasiswa farmasi/biologi/kedokteran."
        ),
        "Umum": (
            "Gunakan bahasa sehari-hari yang sederhana dan mudah dipahami. "
            "Hindari jargon teknis. Fokus pada manfaat praktis dan "
            "pengetahuan umum tentang tanaman obat/herbal."
        ),
    }
    persona_instruction = persona_map.get(ai_mode, persona_map["Pelajar"])

    # Data source adaptation instruction
    if graph_available:
        source_instruction = (
            "Data konteks berasal dari pencarian semantik DAN relasi graph database. "
            "Manfaatkan kedua sumber untuk membuat soal yang kaya dan mendalam."
        )
    else:
        source_instruction = (
            "CATATAN: Data relasi graph tidak tersedia saat ini. "
            "Konteks yang disediakan berasal SEPENUHNYA dari pencarian teks/semantik. "
            "Adaptasi soal berdasarkan informasi teks yang tersedia — "
            "fokus pada fakta, deskripsi, dan konsep yang disebutkan dalam data. "
            "Tetap buat soal berkualitas tinggi meskipun tanpa data relasi antar-entitas."
        )

    return f"""Anda adalah Sistem Pembuat Kuis Farmasi & Kimia yang ketat dan akurat.
Target pengguna: {ai_mode}.

═══ ADAPTASI PERSONA ═══
{persona_instruction}

═══ SUMBER DATA ═══
{source_instruction}

═══ {scope_instruction} ═══

═══ INSTRUKSI MUTLAK ═══
1. Buat TEPAT {jumlah_soal} soal berdasarkan [DATA DATABASE] di bawah.
2. HANYA gunakan informasi dari data yang disediakan sebagai basis soal.
3. JANGAN mengarang informasi ilmiah yang tidak ada dalam data.
4. CRITICAL: Every question MUST have exactly 4 multiple-choice options mapped cleanly to prefixes A, B, C, and D. Never output more or fewer than 4 options.
5. Pembahasan harus merujuk pada data database, bukan pengetahuan umum.
6. Variasikan tingkat kesulitan: Mudah, Menengah, dan HOTS.
7. Setiap opsi jawaban yang salah (distraktor) harus masuk akal, bukan jelas salah.
8. Format id_soal sebagai "Q-01", "Q-02", dst.
9. EVALUATION MANDATE: Fill the `penjelasan_salah` property with a concise sentence explaining why alternative options fail. Fill the root-level `analisis_performa.sorotan` and `area_fokus` with deeply academic, high-quality analytical reviews customized to the user's performance target (e.g., Tingkat SMA).

═══ CRITICAL MANDATE ═══
CRITICAL MANDATE: You MUST generate EXACTLY {jumlah_soal} multiple-choice questions. Do not leave the questions array empty under any circumstances. If the provided context text is short or general, utilize your foundational training data regarding Indonesian medicinal plants (like Temulawak, Sambiloto, Kunyit) to fulfill the exact count of {jumlah_soal} questions matching the educational level requested.
{file_instruction}
═══ DATA DATABASE MULAI ═══
{context_data}
═══ DATA DATABASE SELESAI ═══"""


# ═══════════════════════════════════════════
# TOOL SCHEMA BUILDER
# ═══════════════════════════════════════════

def _build_tool_schema() -> list[dict[str, Any]]:
    """
    Membangun definisi tool untuk OpenAI Tool-Calling.

    Returns:
        List berisi satu tool definition berbasis QuizResponse schema.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "render_interactive_quiz",
                "description": (
                    "Merender kuis interaktif dengan soal pilihan ganda, "
                    "jawaban benar, dan pembahasan langkah demi langkah."
                ),
                "parameters": QuizResponse.model_json_schema(),
            },
        }
    ]


# ═══════════════════════════════════════════
# [4] MULTI-LAYER PARSING & SYNTHETIC FALLBACK
# ═══════════════════════════════════════════

def _parse_tool_calls(message: Any) -> Optional[dict[str, Any]]:
    """
    Check 1: Parse formal API tool_calls dari response LLM.

    Args:
        message: Message object dari LLM response.

    Returns:
        Dict parsed arguments jika berhasil, None jika tidak ada tool_calls.
    """
    if not hasattr(message, "tool_calls") or not message.tool_calls:
        return None

    tool_call = message.tool_calls[0]

    if tool_call.function.name != "render_interactive_quiz":
        logger.warning(
            f"Unexpected tool call: '{tool_call.function.name}' "
            f"(expected 'render_interactive_quiz')."
        )
        return None

    try:
        raw = json.loads(tool_call.function.arguments)
        logger.info("Parsed quiz from tool_calls successfully.")

        # ── ZERO-QUESTION INTERCEPTOR (Defense Layer 1) ──
        # Catch empty questions at the earliest parsing stage
        questions_list = raw.get("daftar_soal", []) or raw.get("questions", [])

        if not questions_list or len(questions_list) == 0:
            logger.warning(
                "LLM returned 0 questions via tool_calls. "
                "Overriding with premium synthetic fallback questions."
            )
            raw["daftar_soal"] = [
                {
                    "id_soal": "Q1",
                    "tingkat_kesulitan": "Umum",
                    "pertanyaan": (
                        "Tanaman obat asli Indonesia yang kaya akan senyawa kurkuminoid "
                        "dan berfungsi efektif sebagai pelindung sel hati (hepatoprotektor) adalah..."
                    ),
                    "opsi_jawaban": [
                        {"label": "A", "text": "Temulawak"},
                        {"label": "B", "text": "Sambiloto"},
                        {"label": "C", "text": "Mahkota Dewa"},
                        {"label": "D", "text": "Daun Meniran"},
                    ],
                    "jawaban_benar": "A",
                    "pembahasan": [
                        "Temulawak (Curcuma zanthorrhiza) terbukti secara klinis "
                        "mengandung kurkumin yang melindungi organ hati dari toksin."
                    ],
                    "penjelasan_salah": "Sambiloto, Mahkota Dewa, dan Daun Meniran tidak dikenal secara utama memiliki kandungan kurkuminoid sebagai pelindung hati.",
                },
                {
                    "id_soal": "Q2",
                    "tingkat_kesulitan": "Umum",
                    "pertanyaan": (
                        "Zat marker aktif utama pada tanaman Sambiloto yang bertanggung jawab "
                        "terhadap stimulasi sistem imun (imunomodulator) adalah..."
                    ),
                    "opsi_jawaban": [
                        {"label": "A", "text": "Alisin"},
                        {"label": "B", "text": "Andrographolide"},
                        {"label": "C", "text": "Piperin"},
                        {"label": "D", "text": "Gidrosida"},
                    ],
                    "jawaban_benar": "B",
                    "pembahasan": [
                        "Andrographolide adalah senyawa aktif pahit pada Sambiloto "
                        "yang bertindak meningkatkan imunitas tubuh."
                    ],
                    "penjelasan_salah": "Alisin, Piperin, dan Gidrosida bukan merupakan zat marker aktif utama pembawa imunomodulator pada Sambiloto.",
                },
            ]
            if "analisis_performa" not in raw:
                raw["analisis_performa"] = {
                    "sorotan": [
                        "Mengidentifikasi khasiat tanaman obat Temulawak secara akurat.",
                        "Memahami senyawa marker aktif imunomodulator pada Sambiloto."
                    ],
                    "area_fokus": [
                        "Meningkatkan pemahaman mengenai spesies tumbuhan obat lainnya.",
                        "Perlu mempelajari klasifikasi senyawa fitokimia secara komprehensif."
                    ]
                }
            logger.info(
                f"Injected {len(raw['daftar_soal'])} fallback questions at parsing stage."
            )

        return raw
    except json.JSONDecodeError as e:
        logger.warning(f"tool_call JSON parse failed: {e}")
        return None


def _parse_content_regex(message: Any) -> Optional[dict[str, Any]]:
    """
    Check 2: Extract JSON dari content string menggunakan regex.

    Handles:
    - JSON di dalam markdown code blocks (```json ... ```)
    - JSON object langsung di dalam content string

    Args:
        message: Message object dari LLM response.

    Returns:
        Dict parsed JSON jika berhasil, None jika tidak ditemukan.
    """
    content = getattr(message, "content", None)
    if not content:
        return None

    # Try markdown code block first
    md_match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", content)
    if md_match:
        try:
            raw = json.loads(md_match.group(1))
            logger.info("Parsed quiz from markdown code block via regex.")
            return raw
        except json.JSONDecodeError:
            pass

    # Try raw JSON object in content
    json_match = re.search(r"(\{[\s\S]*\"daftar_soal\"[\s\S]*\})", content)
    if json_match:
        try:
            raw = json.loads(json_match.group(1))
            logger.info("Parsed quiz from raw JSON in content via regex.")
            return raw
        except json.JSONDecodeError:
            pass

    logger.warning("Regex JSON extraction failed on content.")
    return None


def _generate_synthetic_quiz(topic: str, jumlah_soal: int) -> dict[str, Any]:
    """
    Check 3 (Ultimate Fail-Safe): Local Synthetic Quiz Generator.

    Programmatically builds a valid quiz JSON payload from standardized
    templates when all LLM parsing methods fail. Guarantees the user
    always receives a functional quiz UI.

    Args:
        topic: Topik kuis yang diminta (sudah dibersihkan).
        jumlah_soal: Jumlah soal (clamped ke max 3 untuk synthetic).

    Returns:
        Dict sesuai QuizResponse schema.
    """
    logger.warning(
        f"SYNTHETIC FALLBACK TRIGGERED for topic='{topic[:50]}'. "
        "LLM output was unparsable — generating template quiz."
    )

    # Standardized template questions about medicinal plants & chemistry
    _TEMPLATES: list[dict[str, Any]] = [
        {
            "id_soal": "Q-01",
            "tingkat_kesulitan": "Mudah",
            "pertanyaan": (
                f"Manakah dari berikut ini yang paling tepat menggambarkan "
                f"bidang studi terkait \"{topic}\"?"
            ),
            "opsi_jawaban": [
                {"label": "A", "text": "Studi tentang senyawa kimia aktif dalam tumbuhan obat dan khasiatnya."},
                {"label": "B", "text": "Studi tentang teknik pertanian modern untuk tanaman pangan."},
                {"label": "C", "text": "Studi tentang genetika molekuler hewan vertebrata."},
                {"label": "D", "text": "Studi tentang teknologi pengolahan makanan industri."},
            ],
            "jawaban_benar": "A",
            "pembahasan": [
                f"Topik \"{topic}\" berkaitan erat dengan ilmu farmakognosi and fitokimia.",
                "Bidang ini mempelajari senyawa kimia aktif (metabolit sekunder) dalam tumbuhan obat.",
                "Metabolit sekunder meliputi alkaloid, flavonoid, terpenoid, dan saponin yang memiliki aktivitas farmakologis.",
            ],
            "penjelasan_salah": "Opsi B, C, dan D salah karena teknik pertanian modern, genetika vertebrata, dan pengolahan pangan bukan merupakan fokus utama studi fitokimia dan farmakognosi.",
        },
        {
            "id_soal": "Q-02",
            "tingkat_kesulitan": "Menengah",
            "pertanyaan": (
                "Golongan senyawa metabolit sekunder manakah yang paling dikenal "
                "memiliki aktivitas antioksidan kuat pada tanaman obat?"
            ),
            "opsi_jawaban": [
                {"label": "A", "text": "Flavonoid"},
                {"label": "B", "text": "Asam lemak jenuh"},
                {"label": "C", "text": "Protein struktural"},
                {"label": "D", "text": "Karbohidrat sederhana"},
            ],
            "jawaban_benar": "A",
            "pembahasan": [
                "Flavonoid adalah golongan polifenol yang banyak ditemukan pada tanaman obat.",
                "Senyawa ini memiliki gugus hidroksil (-OH) yang mampu mendonorkan elektron kepada radikal bebas.",
                "Mekanisme ini menjadikan flavonoid sebagai antioksidan alami yang potensial.",
            ],
            "penjelasan_salah": "Opsi B, C, dan D salah karena asam lemak jenuh, protein struktural, dan karbohidrat sederhana tidak dikenal memiliki sifat antioksidan kuat seperti flavonoid.",
        },
        {
            "id_soal": "Q-03",
            "tingkat_kesulitan": "HOTS",
            "pertanyaan": (
                "Seorang pasien mengonsumsi obat pengencer darah (warfarin) "
                "dan ingin menggunakan herbal yang mengandung senyawa kumarin. "
                "Apa risiko utama interaksi yang mungkin terjadi?"
            ),
            "opsi_jawaban": [
                {"label": "A", "text": "Peningkatan efek antikoagulan yang dapat menyebabkan perdarahan."},
                {"label": "B", "text": "Penurunan tekanan darah secara drastis."},
                {"label": "C", "text": "Reaksi alergi berupa ruam kulit."},
                {"label": "D", "text": "Gangguan pencernaan ringan yang bersifat sementara."},
            ],
            "jawaban_benar": "A",
            "pembahasan": [
                "Kumarin dan warfarin sama-sama memiliki efek antikoagulan.",
                "Penggunaan bersamaan dapat menyebabkan sinergisme yang meningkatkan risiko perdarahan.",
                "Ini adalah contoh interaksi obat-herbal yang harus diwaspadai oleh tenaga kesehatan.",
                "Pasien yang mengonsumsi antikoagulan harus berkonsultasi sebelum menggunakan herbal.",
            ],
            "penjelasan_salah": "Opsi B, C, dan D salah karena efek sinergis dengan kumarin berfokus pada antikoagulan/perdarahan, bukan pada tekanan darah, ruam kulit, atau gangguan pencernaan.",
        },
    ]

    # Select the requested number of questions
    selected = _TEMPLATES[:min(jumlah_soal, len(_TEMPLATES))]

    # ── Internal Pydantic validation (uses daftar_soal schema) ──
    analisis_performa_data = {
        "sorotan": [
            f"Memahami konsep dasar fitokimia dan farmakognosi terkait {topic}.",
            "Mampu mengidentifikasi golongan senyawa aktif dan potensi aktivitas biologisnya."
        ],
        "area_fokus": [
            "Perlu memperdalam mekanisme interaksi obat-herbal secara spesifik.",
            "Meningkatkan pemahaman mengenai teknik isolasi dan karakterisasi senyawa aktif."
        ]
    }
    internal_quiz = {
        "topik": topic if topic else "Tanaman Obat & Kimia Farmasi",
        "daftar_soal": selected,
        "analisis_performa": analisis_performa_data,
    }

    try:
        validated = QuizResponse.model_validate(internal_quiz)
        logger.info(
            f"Synthetic quiz generated: {len(validated.daftar_soal)} soal, "
            f"topik='{validated.topik}'."
        )
    except Exception as e:
        logger.error(
            f"Synthetic quiz Pydantic validation failed (critical): {e}",
            exc_info=True,
        )
        validated = None

    # ── Build frontend-aligned payload with `questions` key ──
    # The frontend expects:
    #   { topik, questions: [{ question, options, answer, explanation, penjelasan_salah }] }
    # where options are "A. text" strings and answer is the full "A. text".
    source = validated.daftar_soal if validated else selected
    frontend_questions: list[dict[str, Any]] = []
    for item in source:
        q = item if isinstance(item, dict) else item.model_dump()
        # Build "A. text" style options list
        options_raw = q.get("opsi_jawaban", [])
        labeled_options: list[str] = []
        for opt in options_raw:
            if isinstance(opt, dict):
                labeled_options.append(f"{opt['label']}. {opt['text']}")
            else:
                labeled_options.append(f"{opt.label}. {opt.text}")

        # Resolve full answer string (e.g. "A. Menurunkan demam")
        correct_label = q.get("jawaban_benar", "A")
        full_answer = correct_label
        for opt_str in labeled_options:
            if opt_str.startswith(f"{correct_label}."):
                full_answer = opt_str
                break

        # Flatten pembahasan list into a single explanation string
        pembahasan = q.get("pembahasan", [])
        if isinstance(pembahasan, list):
            explanation = " ".join(pembahasan)
        else:
            explanation = str(pembahasan)

        frontend_questions.append({
            "question": q.get("pertanyaan", ""),
            "options": labeled_options,
            "answer": full_answer,
            "explanation": explanation,
            "penjelasan_salah": q.get("penjelasan_salah", ""),
        })

    return {
        "topik": topic if topic else "Tanaman Obat & Kimia Farmasi",
        "daftar_soal": [q.model_dump() if not isinstance(q, dict) else q for q in source],
        "analisis_performa": analisis_performa_data,
        "questions": frontend_questions,
    }


# ═══════════════════════════════════════════
# MAIN PUBLIC FUNCTION
# ═══════════════════════════════════════════

def generate_interactive_quiz_tool(
    topic: str,
    jumlah_soal: int = 3,
    ai_mode: str = "Pelajar",
    file_context: Optional[str] = None,
) -> dict[str, Any]:
    """
    Generate kuis interaktif — robust, fault-tolerant, never-crash pipeline.

    Pipeline:
    1. NLP Preprocessing: Clean topic, extract jumlah_soal from raw prompt.
    2. Adaptive Retrieval: Specific vs. broad context fetching.
    3. Scope-Aware Prompt: Dynamic system prompt based on topic scope.
    4. LLM Call: Tool-calling with forced schema.
    5. Multi-Layer Parse: tool_calls -> regex -> synthetic fallback.
    6. Pydantic Validation: Final schema check.

    Args:
        topic: Raw topic/prompt dari pengguna.
        jumlah_soal: Jumlah soal default (override jika terdeteksi di prompt).
        ai_mode: Persona AI target (default: Pelajar).
        file_context: Teks dari file upload pengguna (opsional).

    Returns:
        Dict berisi quiz data sesuai QuizResponse schema.
        NEVER raises — always returns a valid quiz payload.
    """
    logger.info(
        f"Quiz pipeline started: raw_topic='{topic[:60]}', "
        f"jumlah_soal={jumlah_soal}, mode={ai_mode}"
    )

    # ── Step 1: NLP Preprocessing ──
    extracted_jumlah = _extract_jumlah_soal(topic, default=jumlah_soal)
    cleaned_topic = _clean_topic(topic)
    jumlah_soal = extracted_jumlah

    logger.info(
        f"After NLP preprocessing: topic='{cleaned_topic}', "
        f"jumlah_soal={jumlah_soal}"
    )

    # ── Step 2: Adaptive Hybrid Retrieval ──
    try:
        context_data, is_broad, graph_available = _retrieve_quiz_context(cleaned_topic)
    except Exception as e:
        logger.error(
            f"Retrieval pipeline crashed: {e}. "
            "Falling back to general corpus.",
            exc_info=True,
        )
        domain = _detect_broad_domain(cleaned_topic)
        corpus_key = domain if domain else "tanaman_obat"
        context_data = _GENERAL_CORPUS.get(corpus_key, _GENERAL_CORPUS["tanaman_obat"])
        is_broad = True
        graph_available = False

    # ── Step 2b: Hard String Truncation ──
    # Cap context to max 4000 chars to guarantee the total request input
    # (system prompt + user message + tool schema) never approaches the
    # model's strict 8193 token boundary.
    _MAX_CONTEXT_CHARS = 4000
    if len(context_data) > _MAX_CONTEXT_CHARS:
        logger.warning(
            f"Context too long ({len(context_data)} chars), "
            f"truncating to {_MAX_CONTEXT_CHARS} chars to avoid "
            f"exceeding model context window."
        )
        context_data = context_data[:4000]

    logger.info(
        f"Retrieval complete: is_broad={is_broad}, "
        f"graph_available={graph_available}, "
        f"context_length={len(context_data)} chars."
    )

    # ── Step 3: Scope-Aware System Prompt ──
    system_prompt = _build_quiz_system_prompt(
        context_data=context_data,
        jumlah_soal=jumlah_soal,
        ai_mode=ai_mode,
        is_broad=is_broad,
        topic=cleaned_topic,
        graph_available=graph_available,
        file_context=file_context,
    )
    tools = _build_tool_schema()

    # ── Step 4: LLM Call ──
    try:
        response = _client.chat.completions.create(
            model=settings.LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Buat kuis {jumlah_soal} soal tentang: {cleaned_topic}"},
            ],
            tools=tools,
            tool_choice={
                "type": "function",
                "function": {"name": "render_interactive_quiz"},
            },
            temperature=0.2,
            max_tokens=1200,
        )
    except Exception as e:
        logger.error(
            f"LLM API call failed: {e}. Triggering synthetic fallback.",
            exc_info=True,
        )
        return _generate_synthetic_quiz(cleaned_topic, jumlah_soal)

    # ── Step 5: Multi-Layer Parsing ──
    message = response.choices[0].message

    # Check 1: Formal tool_calls
    raw_arguments = _parse_tool_calls(message)

    # Check 2: Regex JSON extraction from content
    if raw_arguments is None:
        logger.info("tool_calls parse failed/empty, trying regex extraction...")
        raw_arguments = _parse_content_regex(message)

    # Check 3: Synthetic fallback
    if raw_arguments is None:
        logger.warning(
            "All LLM parsing methods failed. Triggering synthetic fallback."
        )
        return _generate_synthetic_quiz(cleaned_topic, jumlah_soal)

    # ── Step 5b: HARD PYTHON INTERCEPTOR FOR ZERO QUESTIONS ──
    # Prevents empty quiz payloads from reaching validation/frontend.
    # Directly injects high-quality preset questions if LLM returned 0 questions.
    if (
        not isinstance(raw_arguments, dict)
        or not raw_arguments.get("daftar_soal")
        or len(raw_arguments.get("daftar_soal", [])) == 0
    ):
        logger.warning(
            f"ZERO-QUESTION INTERCEPTOR TRIGGERED: LLM returned empty daftar_soal. "
            f"Injecting 3 preset educational questions about Indonesian herbs."
        )
        # Hard override with educational preset questions
        if not isinstance(raw_arguments, dict):
            raw_arguments = {}
        raw_arguments["topik"] = raw_arguments.get("topik", cleaned_topic)
        raw_arguments["daftar_soal"] = [
            {
                "id_soal": "S1",
                "tingkat_kesulitan": "Umum",
                "pertanyaan": (
                    "Tanaman obat herbal Indonesia yang terkenal memiliki kandungan "
                    "senyawa kurkumin dan berfungsi menjaga fungsi hati (hepatoprotektor) adalah..."
                ),
                "opsi_jawaban": [
                    {"label": "A", "text": "Temulawak (Curcuma xanthorrhiza)"},
                    {"label": "B", "text": "Sambiloto (Andrographis paniculata)"},
                    {"label": "C", "text": "Mahkota Dewa (Phaleria macrocarpa)"},
                    {"label": "D", "text": "Daun Sirih (Piper betle)"},
                ],
                "jawaban_benar": "A",
                "pembahasan": [
                    "Temulawak (Curcuma xanthorrhiza) kaya akan senyawa kurkuminoid "
                    "yang terbukti secara klinis mampu melindungi sel hati dari kerusakan.",
                    "Kurkumin bekerja dengan meningkatkan aktivitas enzim antioksidan "
                    "dan menghambat produksi radikal bebas yang merusak hepatosit.",
                ],
                "penjelasan_salah": "Sambiloto, Mahkota Dewa, dan Daun Sirih tidak dikenal secara utama mengandung kurkuminoid sebagai pelindung hati.",
            },
            {
                "id_soal": "S2",
                "tingkat_kesulitan": "Umum",
                "pertanyaan": (
                    "Senyawa aktif utama yang memberikan rasa pahit ekstrim sekaligus "
                    "khasiat imunomodulator pada tanaman Sambiloto adalah..."
                ),
                "opsi_jawaban": [
                    {"label": "A", "text": "Piperin"},
                    {"label": "B", "text": "Andrographolide"},
                    {"label": "C", "text": "Alisin"},
                    {"label": "D", "text": "Kurkumin"},
                ],
                "jawaban_benar": "B",
                "pembahasan": [
                    "Andrographolide merupakan senyawa lakton diterpenoid yang menjadi "
                    "marker utama khasiat medis pada tanaman Sambiloto.",
                    "Senyawa ini memodulasi sistem imun dengan meningkatkan proliferasi "
                    "limfosit T dan aktivitas sel NK (Natural Killer).",
                ],
                "penjelasan_salah": "Piperin, Alisin, dan Kurkumin bukan merupakan marker aktif pahit utama pembawa khasiat imunomodulator pada Sambiloto.",
            },
            {
                "id_soal": "S3",
                "tingkat_kesulitan": "Menengah",
                "pertanyaan": (
                    "Metode ekstraksi yang paling umum digunakan untuk mengekstrak "
                    "senyawa bioaktif polar (seperti flavonoid glikosida) dari tanaman obat adalah..."
                ),
                "opsi_jawaban": [
                    {"label": "A", "text": "Maserasi dengan n-heksana"},
                    {"label": "B", "text": "Maserasi dengan etanol 70%"},
                    {"label": "C", "text": "Destilasi uap"},
                    {"label": "D", "text": "Perkolasi dengan petroleum eter"},
                ],
                "jawaban_benar": "B",
                "pembahasan": [
                    "Etanol 70% merupakan pelarut semi-polar yang efektif mengekstrak "
                    "senyawa polar seperti flavonoid glikosida, tanin, dan saponin.",
                    "Konsentrasi 70% dipilih karena kandungan air membantu penetrasi ke "
                    "dalam sel tumbuhan dan meningkatkan kelarutan senyawa polar.",
                ],
                "penjelasan_salah": "Maserasi dengan n-heksana, destilasi uap, dan perkolasi dengan petroleum eter digunakan untuk ekstraksi senyawa non-polar atau minyak atsiri.",
            },
        ]
        raw_arguments["analisis_performa"] = {
            "sorotan": [
                "Memahami khasiat hepatoprotektor Temulawak.",
                "Mengidentifikasi marker aktif imunomodulator Sambiloto."
            ],
            "area_fokus": [
                "Perlu mempelajari metode ekstraksi senyawa polar dan non-polar lebih lanjut.",
                "Disarankan memahami perbedaan senyawa penanda (marker) pada simplisia herbal."
            ]
        }
        logger.info(
            f"Injected {len(raw_arguments['daftar_soal'])} preset questions "
            f"for topic '{cleaned_topic[:40]}'"
        )

    # ── Step 5c: Self-Healing Defaults for Missing Schema Fields ──
    if isinstance(raw_arguments, dict):
        if "analisis_performa" not in raw_arguments:
            raw_arguments["analisis_performa"] = {
                "sorotan": [
                    f"Memahami konsep materi kuis tentang {cleaned_topic}.",
                    "Menunjukkan pemahaman dasar tentang tanaman obat dan senyawa aktif."
                ],
                "area_fokus": [
                    "Perlu memperdalam detail spesifik dari materi kuis.",
                    "Disarankan membaca kembali referensi pustaka terkait."
                ]
            }
        if "daftar_soal" in raw_arguments and isinstance(raw_arguments["daftar_soal"], list):
            for q in raw_arguments["daftar_soal"]:
                if isinstance(q, dict) and "penjelasan_salah" not in q:
                    q["penjelasan_salah"] = "Opsi lainnya kurang sesuai dengan konteks atau fakta ilmiah yang ditanyakan."

    # ── Step 6: Pydantic Validation ──
    try:
        validated_quiz = QuizResponse.model_validate(raw_arguments)

        # ── Step 6b: Empty Questions Array Guard ──
        # If LLM returned a valid structure but with 0 questions (empty
        # daftar_soal), intercept and fall back to synthetic generation
        # instead of sending a blank payload to the frontend.
        if not validated_quiz.daftar_soal or len(validated_quiz.daftar_soal) == 0:
            logger.warning(
                "LLM returned valid JSON but with 0 questions "
                "(empty daftar_soal). Triggering synthetic fallback."
            )
            return _generate_synthetic_quiz(cleaned_topic, jumlah_soal)

        logger.info(
            f"Quiz generated successfully: "
            f"{len(validated_quiz.daftar_soal)} soal, "
            f"topik='{validated_quiz.topik}', "
            f"scope={'broad' if is_broad else 'specific'}."
        )
        return validated_quiz.model_dump()
    except Exception as e:
        logger.warning(
            f"Pydantic validation failed: {e}. "
            "Checking if raw JSON is usable...",
            exc_info=True,
        )
        # If raw JSON has the minimum required structure, clean and return it
        if (
            isinstance(raw_arguments, dict)
            and "daftar_soal" in raw_arguments
            and isinstance(raw_arguments["daftar_soal"], list)
            and len(raw_arguments["daftar_soal"]) > 0
        ):
            # ── Filter out malformed question items ──
            # Drop any question that lacks critical fields or has empty pertanyaan
            questions_list = raw_arguments["daftar_soal"]
            cleaned_questions = [
                q for q in questions_list
                if isinstance(q, dict)
                and q.get("pertanyaan")
                and isinstance(q.get("pertanyaan"), str)
                and q.get("pertanyaan").strip()
                and q.get("id_soal")
                and q.get("opsi_jawaban")
            ]

            # Check if we have at least 3 valid questions after filtering
            if len(cleaned_questions) < 3:
                logger.warning(
                    f"After filtering malformed questions, only {len(cleaned_questions)} "
                    f"valid questions remain (minimum required: 3). "
                    "Triggering synthetic fallback."
                )
                return _generate_synthetic_quiz(cleaned_topic, jumlah_soal)

            # Update raw_arguments with cleaned questions
            raw_arguments["daftar_soal"] = cleaned_questions

            logger.info(
                f"Returning cleaned raw quiz data with {len(cleaned_questions)} "
                f"valid questions (filtered from {len(questions_list)} raw items)."
            )

            # Ensure topik field exists
            if "topik" not in raw_arguments:
                raw_arguments["topik"] = cleaned_topic

            return raw_arguments

        # Absolute last resort
        logger.warning(
            "Raw JSON structure is also broken. "
            "Triggering synthetic fallback as last resort."
        )
        return _generate_synthetic_quiz(cleaned_topic, jumlah_soal)
