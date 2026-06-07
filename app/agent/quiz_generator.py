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


def _extract_jumlah_soal(raw_prompt: str, default: int = 5) -> int:
    """
    Mengekstrak jumlah soal yang diminta dari prompt pengguna.
    """
    match = _JUMLAH_PATTERN.search(raw_prompt)
    if not match:
        match = _JUMLAH_PATTERN_ALT.search(raw_prompt)
    if match:
        count = int(match.group(1))
    else:
        count = default
    clamped = max(1, min(count, 10))
    logger.info(
        f"Extracted jumlah_soal={clamped} from prompt (raw={count}, clamped={clamped})."
    )
    return clamped


def _clean_topic(raw_prompt: str) -> str:
    """
    Membersihkan noise percakapan dari prompt dan mengekstrak topik inti.
    """
    cleaned = raw_prompt.strip()

    # Remove number-of-questions phrases before general cleaning
    cleaned = re.sub(r'(\d+)\s*(?:soal|pertanyaan)', "", cleaned, flags=re.IGNORECASE)
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
    try:
        query_embedding = embed_text(query)

        result = supabase.rpc("match_education", {
            "query_embedding": query_embedding,
            "match_count": limit,
            "match_threshold": threshold,
        }).execute()

        if result.data:
            logger.info(f"Quiz vector search (education): {len(result.data)} results.")
            return result.data

        result = supabase.rpc("match_plants", {
            "query_embedding": query_embedding,
            "match_count": limit,
            "match_threshold": threshold,
        }).execute()

        if result.data:
            logger.info(f"Quiz vector search (plants): {len(result.data)} results.")
            return result.data

        return []
    except Exception as e:
        logger.error(f"Quiz vector search error: {e}", exc_info=True)
        return []


def _graph_search_quiz(topic: str) -> list[dict[str, Any]]:
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
        logger.error(f"Quiz graph search — Neo4j query failed: {e}", exc_info=True)
        return []


def _broad_graph_search(domain: str) -> list[dict[str, Any]]:
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
        logger.info(f"Broad graph search for domain '{domain}': {len(result)} records.")
        return result
    except Exception as e:
        logger.error(f"Broad graph search failed: {e}", exc_info=True)
        return []


def _format_records(records: list[dict[str, Any]], source_label: str) -> str:
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
    vector_results = _vector_search_quiz(topic, limit=5, threshold=0.7)
    graph_results = _graph_search_quiz(topic)

    vector_text = _format_records(vector_results, "Pencarian Semantik")
    graph_text = _format_records(graph_results, "Relasi Graph Database")

    graph_available = bool(graph_results)
    has_specific_data = bool(vector_results) or graph_available

    if has_specific_data:
        context = "\n\n".join(filter(None, [vector_text, graph_text]))
        return context, False, graph_available

    logger.info(f"No specific results for '{topic[:40]}', trying threshold (0.4)...")
    vector_results_low = _vector_search_quiz(topic, limit=8, threshold=0.4)
    vector_text_low = _format_records(vector_results_low, "Pencarian Semantik (Diperluas)")

    if vector_results_low:
        broad_graph = _broad_graph_search("general")
        broad_graph_text = _format_records(broad_graph, "Data Umum Database")
        context = "\n\n".join(filter(None, [vector_text_low, broad_graph_text]))
        return context, True, bool(broad_graph)

    logger.info(f"Vector search empty for '{topic[:40]}', falling back to corpus...")
    domain = _detect_broad_domain(topic)
    broad_graph = _broad_graph_search(domain or "general")
    broad_graph_text = _format_records(broad_graph, "Data Umum Database")

    if domain and domain in _GENERAL_CORPUS:
        corpus_text = f"[Ringkasan Pengetahuan Domain '{domain.upper()}']:\n{_GENERAL_CORPUS[domain]}"
    else:
        all_corpus = "\n\n".join(f"[{k.upper()}]: {v}" for k, v in _GENERAL_CORPUS.items())
        corpus_text = f"[Ringkasan Pengetahuan Umum Farmasi & Tanaman Obat]:\n{all_corpus}"

    context = "\n\n".join(filter(None, [broad_graph_text, corpus_text]))
    return context, True, bool(broad_graph)


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
    file_instruction = ""
    if file_context:
        safe_context = file_context[:3500]
        file_instruction = f"""
═══ DATA KONTEKS FILE YANG DIUNGGAH USER (PRIORITAS TERTINGGI) ═══
{safe_context}
═══ AKHIR DATA FILE ═══

CRITICAL DIRECTIVE:
1. User telah melampirkan berkas dokumen/gambar medis. WAJIB prioritaskan konten file ini saat membuat soal kuis.
2. Buat pertanyaan yang langsung menguji pemahaman konsep, fakta, dan informasi dari file di atas.
3. JANGAN buat soal generik yang tidak terkait dengan isi file.
4. Fokus pada senyawa, mekanisme, dan data spesifik yang ada dalam file.
5. Lengkapi dengan database hanya jika konten file tidak cukup untuk jumlah soal yang diminta."""

    if is_broad:
        scope_instruction = (
            "SCOPE: TOPIK UMUM/GENERAL.\n"
            "- Buat soal KONSEPTUAL dan FUNDAMENTAL yang mencakup prinsip dasar dari domain yang diminta.\n"
            "- Fokus pada pemahaman definisi, klasifikasi, fungsi umum, dan perbandingan antar konsep.\n"
            "- Soal harus bervariasi: definisi, perbandingan, sebab-akibat, dan penerapan.\n"
            f"- Topik utama yang diminta pengguna: \"{topic}\"."
        )
    else:
        scope_instruction = (
            "SCOPE: TOPIK SPESIFIK.\n"
            "- Buat soal MENDALAM dan CASE-STUDY style yang fokus pada mekanisme, senyawa spesifik, dan interaksi klinis.\n"
            "- Sertakan detail seperti nama senyawa aktif, mekanisme aksi, efek farmakologis, dan interaksi obat.\n"
            f"- Topik spesifik: \"{topic}\"."
        )

    persona_map: dict[str, str] = {
        "Tenaga Medis": "Gunakan terminologi klinis dan farmakologis yang presisi. Soal setara level tenaga kesehatan profesional.",
        "Peneliti": "Gunakan bahasa ilmiah formal. Sertakan nama latin, kelas senyawa, dan terminologi metodologis (IC50, GC-MS, HPLC).",
        "Pelajar": "Gunakan bahasa yang edukatif, mudah dipahami, cocok untuk mahasiswa farmasi/biologi/kedokteran/Kimia SMA.",
        "Umum": "Gunakan bahasa sehari-hari yang sederhana dan mudah dipahami. Hindari jargon teknis. Fokus pada manfaat praktis.",
    }
    persona_instruction = persona_map.get(ai_mode, persona_map["Pelajar"])

    if graph_available:
        source_instruction = "Data konteks berasal dari pencarian semantik DAN relasi graph database. Manfaatkan keduanya untuk membuat soal."
    else:
        source_instruction = "Data berasal sepenuhnya dari pencarian teks. Adaptasikan soal berdasarkan informasi teks yang tersedia."

    return f"""Anda adalah Sistem Pembuat Kuis Farmasi & Kimia yang ketat dan akurat.
Target pengguna: {ai_mode}.

═══ ADAPTASI PERSONA ═══
{persona_instruction}

═══ SUMBER DATA ═══
{source_instruction}

═══ {scope_instruction} ═══

═══ ATURAN MUTLAK PEMBUATAN OPSI JAWABAN (DISTRACTORS) ═══
- Pilihan jawaban (A, B, C, D) WAJIB bervariasi, unik, dan dirancang secara ilmiah sesuai dengan topik pertanyaan.
- DILARANG KERAS mengulang-ulang opsi generik yang sama (seperti 'Flavonoid', 'Asam lemak jenuh', 'Protein struktural') di setiap soal!
- Jika pertanyaan membahas tentang tata nama senyawa, maka opsi harus berupa nama senyawa kimia. Jika membahas termokimia, opsi harus berupa nilai entalpi atau jenis reaksi (eksoterm/endoterm).
- Buat pengecoh (distractors) yang cerdas dan masuk akal bagi target pengguna.

═══ INSTRUKSI MUTLAK OUTPUT FORMAT ═══
1. Buat TEPAT {jumlah_soal} soal berdasarkan data di bawah.
2. Setiap pertanyaan WAJIB memiliki tepat 4 pilihan ganda (A, B, C, D).
3. Anda WAJIB langsung mengeluarkan jawaban dalam bentuk struktur objek JSON valid yang mematuhi skema QuizResponse.
4. JANGAN menyertakan teks pembuka, penutup, atau penanda blok kode markdown di luar struktur objek tersebut. Output harus berupa RAW JSON murni yang langsung dapat di-parse.
5. Format id_soal sebagai "Q-01", "Q-02", dst.
6. Isi properti `penjelasan_salah` untuk menerangkan mengapa opsi lain salah. Fill root-level `analisis_performa.sorotan` and `area_fokus` with deeply academic, high-quality analytical reviews.

═══ CRITICAL MANDATE ═══
Anda harus membuat EXACTLY {jumlah_soal} soal pilihan ganda. Jangan biarkan array kosong. Jika konteks database pendek, manfaatkan ilmu dasar fito-farmaka Anda (seperti Ginseng, Temulawak, Kunyit, Sambiloto) untuk melengkapi total {jumlah_soal} soal yang valid.
{file_instruction}
═══ DATA DATABASE MULAI ═══
{context_data}
═══ DATA DATABASE SELESAI ═══"""


# ═══════════════════════════════════════════
# PARSING & UTILITIES LAYER
# ═══════════════════════════════════════════

def _build_tool_schema(jumlah_soal: int) -> list[dict[str, Any]]:
    """
    Membangun definisi tool kustom formal untuk skema QuizResponse.
    """
    schema = QuizResponse.model_json_schema()
    if "properties" in schema and "daftar_soal" in schema["properties"]:
        schema["properties"]["daftar_soal"]["minItems"] = jumlah_soal
        schema["properties"]["daftar_soal"]["maxItems"] = jumlah_soal
    return [
        {
            "type": "function",
            "function": {
                "name": "render_interactive_quiz",
                "description": (
                    "Merender kuis interaktif dengan soal pilihan ganda, "
                    "jawaban benar, dan pembahasan langkah demi langkah."
                ),
                "parameters": schema,
            },
        }
    ]


def _parse_tool_calls(message: Any) -> Optional[dict[str, Any]]:
    """
    Check 1: Parse formal API tool_calls dari response LLM secara aman.
    """
    if not hasattr(message, "tool_calls") or not message.tool_calls:
        return None

    tool_call = message.tool_calls[0]
    if tool_call.function.name != "render_interactive_quiz":
        logger.warning(f"Unexpected tool call function: '{tool_call.function.name}'")
        return None

    try:
        raw = json.loads(tool_call.function.arguments)
        logger.info("Parsed quiz from formal tool_calls parameter block successfully.")
        
        questions_list = raw.get("daftar_soal", []) or raw.get("questions", [])
        if not questions_list or len(questions_list) == 0:
            logger.warning("LLM returned empty questions list via tool_calls. Triggering next layer.")
            return None
            
        return raw
    except json.JSONDecodeError as e:
        logger.warning(f"tool_call JSON parsing error chunk: {e}")
        return None


def _parse_content_regex(message: Any) -> Optional[dict[str, Any]]:
    """
    Check 2: Extract JSON dari content string menggunakan regex.
    """
    content = getattr(message, "content", None)
    if not content:
        return None

    content_clean = content.strip()
    
    # PROTEKSI ANTI-POTONG: Menggunakan escape heksadesimal '\x60' untuk backtick
    # Ini 100% menjamin teks kode program tidak akan merusak markdown formatter luar
    triple_backtick = "\x60\x60\x60"
    if content_clean.startswith(triple_backtick):
        content_clean = re.sub(r"^[\x60]{3}(?:json)?\s*", "", content_clean, flags=re.IGNORECASE)
        content_clean = re.sub(r"\s*[\x60]{3}$", "", content_clean)

    try:
        raw = json.loads(content_clean.strip())
        logger.info("Parsed quiz from clean string content successfully.")
        return raw
    except json.JSONDecodeError:
        pass

    json_match = re.search(r"(\{[\s\S]*\"daftar_soal\"[\s\S]*\})", content)
    if json_match:
        try:
            raw = json.loads(json_match.group(1))
            logger.info("Parsed quiz from nested JSON bracket structure via regex.")
            return raw
        except json.JSONDecodeError:
            pass

    logger.warning("Regex JSON extraction completely failed on content string.")
    return None


def _map_to_frontend_payload(validated_quiz_dict: dict[str, Any], count: int) -> dict[str, Any]:
    source_questions = validated_quiz_dict.get("daftar_soal", [])
    mapped_questions = []
    for i, item in enumerate(source_questions[:count]):
        options_raw = item.get("opsi_jawaban", [])
        options_dict_list = []
        for opt in options_raw:
            if isinstance(opt, dict):
                options_dict_list.append({"label": opt.get("label", ""), "text": opt.get("text", "")})
            else:
                options_dict_list.append({"label": getattr(opt, "label", ""), "text": getattr(opt, "text", "")})
        
        options_str_list = [f"{opt['label']}. {opt['text']}" for opt in options_dict_list]
        correct_label = item.get("jawaban_benar", "A")
        full_answer = correct_label
        for opt_str in options_str_list:
            if opt_str.startswith(f"{correct_label}."):
                full_answer = opt_str
                break
                
        pembahasan = item.get("pembahasan", [])
        explanation = " ".join(pembahasan) if isinstance(pembahasan, list) else str(pembahasan)
        
        mapped_questions.append({
            "question_text": item.get("pertanyaan", ""),
            "question": item.get("pertanyaan", ""),
            "options": options_dict_list,
            "options_labeled": options_str_list,
            "correct_answer": correct_label,
            "answer": full_answer,
            "explanation": explanation,
            "penjelasan_salah": item.get("penjelasan_salah", ""),
            "id_soal": item.get("id_soal", f"Q-{i+1:02d}"),
            "tingkat_kesulitan": item.get("tingkat_kesulitan", "Menengah")
        })
    return {
        "topik": validated_quiz_dict.get("topik", ""),
        "daftar_soal": source_questions,
        "analisis_performa": validated_quiz_dict.get("analisis_performa", {}),
        "questions": mapped_questions
    }


def _generate_synthetic_quiz(topic: str, jumlah_soal: int) -> dict[str, Any]:
    logger.warning(
        f"SYNTHETIC FALLBACK TRIGGERED for topic='{topic[:50]}'. "
        f"Generating completely dynamic {jumlah_soal} template quiz array rows."
    )
    
    _POOL = [
        {
            "tingkat_kesulitan": "Mudah",
            "pertanyaan": f"Manakah dari pernyataan berikut yang paling tepat menggambarkan fokus utama studi fitokimia terkait ekstrak herbal \"{topic}\"?",
            "opsi_jawaban": [
                {"label": "A", "text": "Isolasi dan karakterisasi senyawa metabolit sekunder bioaktif."},
                {"label": "B", "text": "Penghitungan kadar air permukaan tanah perkebunan makro."},
                {"label": "C", "text": "Analisis rantai pasok dan distribusi makro ekonomi global."},
                {"label": "D", "text": "Klasifikasi morfologi luar rumpun kingdom hewan vertebrata."},
            ],
            "jawaban_benar": "A",
            "pembahasan": [
                f"Studi fitokimia pada komoditas \"{topic}\" berfokus pada pemisahan, pemurnian, dan identifikasi molekul aktif (metabolit sekunder) seperti alkaloid, flavonoid, dan saponin.",
            ],
            "penjelasan_salah": "Opsi B, C, dan D salah karena aspek perkebunan, makro ekonomi, dan zoologi bukan bagian dari ruang lingkup analisis fitokimia bahan alam laboratorium."
        },
        {
            "tingkat_kesulitan": "Menengah",
            "pertanyaan": f"Senyawa penanda aktif (marker compound) manakah yang menjadi parameter standardisasi QC klinis utama pada tanaman obat tradisional golongan \"{topic}\"?",
            "opsi_jawaban": [
                {"label": "A", "text": f"Senyawa metabolit sekunder spesifik (seperti flavonoid/glikosida pada \"{topic}\")."},
                {"label": "B", "text": "Glukosa rantai lurus jenuh bebas struktural"},
                {"label": "C", "text": "Asam amino penyusun struktur dinding sel selulosa"},
                {"label": "D", "text": "Kandungan mineral garam natrium klorida eksternal bumi"},
            ],
            "jawaban_benar": "A",
            "pembahasan": [
                f"Standardisasi sediaan herbal \"{topic}\" mewajibkan pengukuran kadar kandungan senyawa penanda (marker) bioaktif untuk menjamin konsistensi efikasi terapeutik antar sediaan.",
            ],
            "penjelasan_salah": "Opsi B, C, dan D salah karena mineral bumi, asam amino selulosa, dan glukosa bebas bersifat umum dan tidak dapat dijadikan penanda unik sediaan fito-farmaka."
        },
        {
            "tingkat_kesulitan": "HOTS",
            "pertanyaan": f"Jika sediaan ekstrak kental tanaman obat \"{topic}\" menunjukkan efek sinergis positif saat dikombinasikan dengan terapia konvensional, parameter farmakokinetik apa yang paling kritis untuk dipantau?",
            "opsi_jawaban": [
                {"label": "A", "text": "Modulasi aktivitas enzim metabolisme sitokrom P450 di organ hepar."},
                {"label": "B", "text": "Peningkatan ekskresi air seni melalui filtrasi glomerulus renal."},
                {"label": "C", "text": "Kecepatan disintegrasi fisik struktur tablet di lambung."},
                {"label": "D", "text": "Perubahan tingkat sensitivitas indra pengecap mukosa lidah."},
            ],
            "jawaban_benar": "A",
            "pembahasan": [
                f"Sinergisme obat-herbal \"{topic}\" sangat dipengaruhi oleh proses ADME. Penghambatan atau induksi enzim sitokrom P450 dapat mengubah kadar plasma obat konvensional secara signifikan.",
            ],
            "penjelasan_salah": "Opsi B, C, dan D kurang tepat karena disintegrasi lambung, indra pengecap, dan filtrasi ginjal bukan merupakan jalur utama modulasi sinergisme farmakokinetik molekular."
        }
    ]

    selected_questions = []
    for i in range(jumlah_soal):
        tpl = _POOL[i % len(_POOL)].copy()
        tpl["id_soal"] = f"Q-{i+1:02d}"
        tpl["opsi_jawaban"] = [opt.copy() for opt in tpl["opsi_jawaban"]]
        selected_questions.append(tpl)

    analisis_performa_data = {
        "sorotan": [
            f"Mampu memetakan ruang lingkup fitokimia dasar komoditas {topic}.",
            "Mengidentifikasi pentingnya penanda aktif (marker) dalam standardisasi simplisia."
        ],
        "area_fokus": [
            "Perlu memperdalam jalur metabolisme ADME dan interaksi enzim sitokrom P450.",
            "Disarankan meninjau kembali regulasi standardisasi sediaan ekstrak herbal."
        ]
    }
    
    internal_quiz = {
        "topik": topic if topic else "Tanaman Obat & Kimia Farmasi",
        "daftar_soal": selected_questions,
        "analisis_performa": analisis_performa_data,
    }

    logger.info(f"Synthetic quiz generated: {len(selected_questions)} soal, topik='{topic}'.")
    return _map_to_frontend_payload(internal_quiz, jumlah_soal)


# ═══════════════════════════════════════════
# MAIN INTERACTIVE QUIZ TOOL ENDPOINT
# ═══════════════════════════════════════════

def generate_interactive_quiz_tool(
    topic: str,
    jumlah_soal: int = 3,
    ai_mode: str = "Pelajar",
    file_context: Optional[str] = None,
    model: Optional[str] = None,
) -> dict[str, Any]:
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
            f"Retrieval pipeline crashed: {e}. Falling back to general corpus.",
            exc_info=True,
        )
        domain = _detect_broad_domain(cleaned_topic)
        corpus_key = domain if domain else "tanaman_obat"
        context_data = _GENERAL_CORPUS.get(corpus_key, _GENERAL_CORPUS["tanaman_obat"])
        is_broad = True
        graph_available = False

    # ── Step 2b: Hard String Truncation ──
    _MAX_CONTEXT_CHARS = 4000
    if len(context_data) > _MAX_CONTEXT_CHARS:
        logger.warning(
            f"Context too long ({len(context_data)} chars), truncating to {_MAX_CONTEXT_CHARS} chars."
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
    tools = _build_tool_schema(jumlah_soal)

    # ── Step 4: Secure LLM Call ──
    resolved_model = model or settings.LLM_DEFAULT_MODEL
    try:
        logger.info(f"Invoking HuggingFace completions API natively with model instance: {resolved_model}")
        response = _client.chat.completions.create(
            model=resolved_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Hasilkan objek JSON kuis murni berisi TEPAT {jumlah_soal} soal tentang materi: {cleaned_topic}. Output harus berupa JSON valid sesuai dengan instruksi system."},
            ],
            temperature=0.2,
            max_tokens=3000,
        )
    except Exception as e:
        logger.error(
            f"LLM API call failed with exception: {e}. Triggering dynamic synthetic fallback.",
            exc_info=True,
        )
        return _generate_synthetic_quiz(cleaned_topic, jumlah_soal)

    # ── Step 5: Multi-Layer Parsing Layer ──
    message = response.choices[0].message
    raw_arguments = _parse_tool_calls(message)

    if raw_arguments is None:
        logger.info("Formal tool_calls failed or returned empty. Running regex parser layer...")
        raw_arguments = _parse_content_regex(message)

    if raw_arguments is None:
        logger.warning("All LLM JSON parsing channels failed. Triggering synthetic fallback.")
        return _generate_synthetic_quiz(cleaned_topic, jumlah_soal)

    # ── Step 5b: HARD PYTHON INTERCEPTOR FOR ZERO QUESTIONS ──
    if (
        not isinstance(raw_arguments, dict)
        or not raw_arguments.get("daftar_soal")
        or len(raw_arguments.get("daftar_soal", [])) == 0
    ):
        logger.warning(
            f"ZERO-QUESTION INTERCEPTOR TRIGGERED. Injecting dynamic {jumlah_soal} preset questions."
        )
        return _generate_synthetic_quiz(cleaned_topic, jumlah_soal)

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
            raw_arguments["daftar_soal"] = raw_arguments["daftar_soal"][:jumlah_soal]
            for q in raw_arguments["daftar_soal"]:
                if isinstance(q, dict) and "penjelasan_salah" not in q:
                    q["penjelasan_salah"] = "Opsi lainnya kurang sesuai dengan konteks atau fakta ilmiah yang ditanyakan."

    # ── Step 6: Final Pydantic Validation & Payload Mapping ──
    try:
        validated_quiz = QuizResponse.model_validate(raw_arguments)
        logger.info(
            f"Quiz generated successfully from HuggingFace pipeline: {len(validated_quiz.daftar_soal)} soal."
        )
        return _map_to_frontend_payload(validated_quiz.model_dump(), jumlah_soal)
    except Exception as e:
        logger.warning(f"Pydantic schema validation failed: {e}. Resolving dictionary raw structures...", exc_info=True)
        if (
            isinstance(raw_arguments, dict)
            and "daftar_soal" in raw_arguments
            and isinstance(raw_arguments["daftar_soal"], list)
            and len(raw_arguments["daftar_soal"]) > 0
        ):
            questions_list = raw_arguments["daftar_soal"]
            cleaned_questions = [
                q for q in questions_list
                if isinstance(q, dict)
                and q.get("pertanyaan")
                and q.get("id_soal")
                and q.get("opsi_jawaban")
            ]

            if len(cleaned_questions) == 0:
                return _generate_synthetic_quiz(cleaned_topic, jumlah_soal)

            raw_arguments["daftar_soal"] = cleaned_questions[:jumlah_soal]
            if "topik" not in raw_arguments:
                raw_arguments["topik"] = cleaned_topic
            return _map_to_frontend_payload(raw_arguments, jumlah_soal)

        return _generate_synthetic_quiz(cleaned_topic, jumlah_soal)