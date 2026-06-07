# -*- coding: utf-8 -*-
"""
Quiz Generator - Agentic Tool-Calling untuk pembuatan kuis interaktif (v2.0.0).
"""

import json
import logging
import re
import os
import secrets
import uuid
import hashlib
from typing import Any, Optional, Literal, List

from huggingface_hub import InferenceClient
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.database import neo4j_driver, supabase
from app.core.embedding import embed_text
from app.models.quiz_schemas import QuizResponse, QuizQuestion, QuizOption

logger = logging.getLogger(__name__)

# Centralized configuration constants
MIN_QUIZ_QUESTIONS = 1
DEFAULT_QUIZ_QUESTIONS = 5
MAX_QUIZ_QUESTIONS = 50
QUIZ_GENERATOR_VERSION = "2.0.0"

# Error classes
class QuizError(Exception):
    pass

class RetrievalError(QuizError):
    pass

class LLMGenerationError(QuizError):
    pass

class QuizParsingError(QuizError):
    pass

class QuizValidationError(QuizError):
    pass

class DuplicateQuestionError(QuizError):
    pass

class IncompleteQuizError(QuizError):
    pass

# LLM CLIENT (Shared, Singleton) - HuggingFace Inference API
_client = InferenceClient(
    provider="auto",
    api_key=settings.HF_API_TOKEN,
)

# In-memory session history cache for deduplication across requests
SESSION_QUIZ_HISTORY: dict[str, list[str]] = {}

# Preprocessing patterns
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

_JUMLAH_PATTERN = re.compile(
    r"(\d+)\s*(?:buah|butir|nomor|nomer)?\s*(?:soal|pertanyaan|kuis|quiz|question)",
    re.IGNORECASE,
)
_JUMLAH_PATTERN_ALT = re.compile(
    r"(?:soal|pertanyaan|kuis|quiz|question)\s*(?:sebanyak)?\s*(\d+)",
    re.IGNORECASE,
)

def _extract_jumlah_soal(raw_prompt: str, default: int = DEFAULT_QUIZ_QUESTIONS) -> int:
    """
    Mengekstrak jumlah soal yang diminta dari prompt pengguna secara akurat.
    """
    match = _JUMLAH_PATTERN.search(raw_prompt)
    if not match:
        match = _JUMLAH_PATTERN_ALT.search(raw_prompt)
    if match:
        count = int(match.group(1))
    else:
        count = default
    clamped = max(MIN_QUIZ_QUESTIONS, min(count, MAX_QUIZ_QUESTIONS))
    logger.info(
        f"Extracted jumlah_soal={clamped} from prompt (raw={count}, clamped={clamped})."
    )
    return clamped

def _clean_topic(raw_prompt: str) -> str:
    """
    Membersihkan noise percakapan dari prompt dan mengekstrak topik inti.
    """
    cleaned = raw_prompt.strip()
    cleaned = re.sub(r"(\d+)\s*(?:soal|pertanyaan)", "", cleaned, flags=re.IGNORECASE)
    cleaned = _JUMLAH_PATTERN.sub("", cleaned)
    cleaned = _JUMLAH_PATTERN_ALT.sub("", cleaned)

    for pattern in _NOISE_PATTERNS:
        cleaned = pattern.sub(" ", cleaned)

    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned or len(cleaned) < 2:
        logger.warning(
            f"Topic cleaning resulted in empty string, using original prompt: \'{raw_prompt[:60]}\'"
        )
        return raw_prompt.strip()

    logger.info(f"Topic cleaned: \'{raw_prompt[:60]}\' -> \'{cleaned}\'")
    return cleaned

# RAG AND SEARCH METHODS
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
        "meliputi: Kunyit (Curcuma longa - kurkumin, antiinflamasi, hepatoprotektor), "
        "Temulawak (Curcuma xanthorrhiza - xanthorrhizol, hepatoprotektor), "
        "Jahe (Zingiber officinale - gingerol, antiemetik, analgesik), "
        "Sambiloto (Andrographis paniculata - andrografolid, immunomodulator), "
        "Mengkudu (Morinda citrifolia - skopoletin, antihipertensi), "
        "Pegagan (Centella asiatica - asiatikosida, penyembuhan luka), dan "
        "Kumis Kucing (Orthosiphon stamineus - sinensetin, diuretik). "
        "Bagian tumbuhan yang digunakan: rhizoma, folium, radix, cortex, flos, fructus, semen."
    ),
    "farmasi": (
        "Farmasi herbal mencakup: Farmakodinamik (mekanisme aksi senyawa pada "
        "reseptor dan enzim), Farmakokinetik (ADME - absorpsi, distribusi, "
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
        embedded_query = embed_text(query)
        rpc_params = {
            "query_embedding": embedded_query,
            "match_threshold": threshold,
            "match_count": limit,
        }
        res = supabase.rpc("match_chunks", rpc_params).execute()
        return res.data or []
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
        logger.info(f"Quiz graph search: {len(result)} records for \'{topic[:40]}\'.")
        return result
    except Exception as e:
        logger.error(f"Quiz graph search -- Neo4j query failed: {e}", exc_info=True)
        return []

def _broad_graph_search(domain: str) -> list[dict[str, Any]]:
    keywords = _BROAD_KEYWORD_MAP.get(domain, ["Herb", "Medical", "Plant"])
    cypher = """
    UNWIND $keywords AS kw
    MATCH (h:Herb)
    WHERE toLower(h.name) CONTAINS toLower(kw)
       OR any(c IN h.concepts WHERE toLower(c) CONTAINS toLower(kw))
    OPTIONAL MATCH (h)-[:HAS_COMPOUND]->(c:Compound)
    OPTIONAL MATCH (h)-[:USED_FOR]->(t:TherapeuticUse)
    RETURN DISTINCT h.name AS topik,
           h.name AS deskripsi,
           collect(DISTINCT c.name) AS konsep_kunci,
           collect(DISTINCT t.name) AS topik_terkait
    LIMIT 8
    """
    try:
        records, _, _ = neo4j_driver.execute_query(
            cypher,
            parameters_={"keywords": keywords}
        )
        result = [record.data() for record in records]
        logger.info(f"Broad graph search for domain \'{domain}\' with keywords {keywords}: {len(result)} records.")
        if not result:
            fallback_cypher = """
            MATCH (h:Herb)
            OPTIONAL MATCH (h)-[:HAS_COMPOUND]->(c:Compound)
            OPTIONAL MATCH (h)-[:USED_FOR]->(t:TherapeuticUse)
            RETURN h.name AS topik,
                   h.name AS deskripsi,
                   collect(DISTINCT c.name) AS konsep_kunci,
                   collect(DISTINCT t.name) AS topik_terkait
            LIMIT 5
            """
            records, _, _ = neo4j_driver.execute_query(fallback_cypher)
            result = [record.data() for record in records]
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

    logger.info(f"No specific results for \'{topic[:40]}\', trying threshold (0.4)...")
    vector_results_low = _vector_search_quiz(topic, limit=8, threshold=0.4)
    vector_text_low = _format_records(vector_results_low, "Pencarian Semantik (Diperluas)")

    if vector_results_low:
        broad_graph = _broad_graph_search("general")
        broad_graph_text = _format_records(broad_graph, "Data Umum Database")
        context = "\n\n".join(filter(None, [vector_text_low, broad_graph_text]))
        return context, True, bool(broad_graph)

    logger.info(f"Vector search empty for \'{topic[:40]}\', falling back to corpus...")
    domain = _detect_broad_domain(topic)
    broad_graph = _broad_graph_search(domain or "general")
    broad_graph_text = _format_records(broad_graph, "Data Umum Database")

    if domain and domain in _GENERAL_CORPUS:
        corpus_text = f"[Ringkasan Pengetahuan Domain \'{domain.upper()}\']:\n{_GENERAL_CORPUS[domain]}"
    else:
        all_corpus = "\n\n".join(f"[{k.upper()}]: {v}" for k, v in _GENERAL_CORPUS.items())
        corpus_text = f"[Ringkasan Pengetahuan Umum Farmasi & Tanaman Obat]:\n{all_corpus}"

    context = "\n\n".join(filter(None, [broad_graph_text, corpus_text]))
    return context, True, bool(broad_graph)

# BLUEPRINT AND SYSTEM PROMPT COMPILER
class QuizBlueprintItem(BaseModel):
    nomor: int
    subtopik: str
    tingkat_kesulitan: Literal["Mudah", "Menengah", "Sulit", "HOTS"]
    level_kognitif: str
    jenis_soal: str
    sudut_pandang: str
    correct_label: Literal["A", "B", "C", "D"]

class QuizBlueprint(BaseModel):
    topik: str
    generation_id: str
    items: list[QuizBlueprintItem]

def create_quiz_blueprint(topic: str, count: int, domain: Optional[str]) -> QuizBlueprint:
    if domain == "kimia":
        subtopics = ["Fitokimia", "Metabolit Sekunder", "Teknik Analisis Kuantitatif", "Uji Aktivitas", "Reaksi Identifikasi"]
        question_types = ["konsep atau definisi", "identifikasi fakta", "perbandingan", "penerapan konsep", "analisis pernyataan", "interpretasi data", "metode laboratorium"]
    elif domain == "tanaman_obat":
        subtopics = ["Kandungan Kimia", "Khasiat Terapeutik", "Keamanan Penggunaan", "Morfologi & Karakter", "Budidaya & QC"]
        question_types = ["konsep atau definisi", "identifikasi fakta", "sebab-akibat", "studi kasus", "evaluasi keamanan", "standardisasi mutu"]
    elif domain == "farmasi":
        subtopics = ["Farmakodinamika", "Farmakokinetika", "Interaksi Obat-Herbal", "Formulasi Sediaan", "Standardisasi Mutu"]
        question_types = ["penerapan konsep", "sebab-akibat", "mekanisme biologis", "interaksi obat", "skenario klinis edukatif", "evaluasi keamanan"]
    elif domain == "herbal":
        subtopics = ["Regulasi BPOM", "Ekstraksi & Pelarut", "Uji Bioaktivitas", "Empiris Tradisional", "Keamanan & Efek Samping"]
        question_types = ["konsep atau definisi", "sebab-akibat", "standardisasi mutu", "metode laboratorium", "evaluasi keamanan", "studi kasus"]
    else:
        subtopics = ["Konseptual Dasar", "Manfaat Praktis", "Struktur & Karakter", "Penerapan Umum", "Studi Kasus"]
        question_types = ["konsep atau definisi", "identifikasi fakta", "sebab-akibat", "perbandingan", "penerapan konsep", "studi kasus", "analisis pernyataan"]

    rng = secrets.SystemRandom()
    base_labels = ["A", "B", "C", "D"]
    labels_pool = []
    for idx in range(count):
        labels_pool.append(base_labels[idx % 4])
    rng.shuffle(labels_pool)

    difficulties = ["Mudah", "Menengah", "Sulit", "HOTS"]
    kognitif_levels = ["Mengingat", "Memahami", "Menerapkan", "Menganalisis", "Mengevaluasi"]
    sudut_pandangs = ["praktis", "klinis/laboratorium", "teoretis", "edukatif"]

    items = []
    for i in range(count):
        item = QuizBlueprintItem(
            nomor=i + 1,
            subtopik=subtopics[i % len(subtopics)],
            tingkat_kesulitan=difficulties[i % len(difficulties)],
            level_kognitif=kognitif_levels[i % len(kognitif_levels)],
            jenis_soal=question_types[i % len(question_types)],
            sudut_pandang=sudut_pandangs[i % len(sudut_pandangs)],
            correct_label=labels_pool[i]
        )
        items.append(item)

    return QuizBlueprint(
        topik=topic,
        generation_id=str(uuid.uuid4()),
        items=items
    )

def _build_quiz_system_prompt(
    context_data: str,
    batch_count: int,
    total_count: int,
    ai_mode: str,
    topic: str,
    generation_nonce: str,
    blueprint_json: str,
    file_context: Optional[str] = None,
) -> str:
    file_instruction = ""
    if file_context:
        safe_context = file_context[:3500]
        file_instruction = f"""
??? DATA KONTEKS FILE YANG DIUNGGAH USER (PRIORITAS TERTINGGI) ???
{safe_context}
??? AKHIR DATA FILE ???

CRITICAL DIRECTIVE:
1. User telah melampirkan berkas dokumen/gambar medis. WAJIB prioritaskan konten file ini saat membuat soal kuis.
2. Buat pertanyaan yang langsung menguji pemahaman konsep, fakta, dan informasi dari file di atas.
3. JANGAN buat soal generik yang tidak terkait dengan isi file.
4. Fokus pada senyawa, mekanisme, dan data spesifik yang ada dalam file.
"""

    persona_map: dict[str, str] = {
        "Tenaga Medis": "Gunakan terminologi klinis dan farmakologis yang presisi. Soal setara level tenaga kesehatan profesional.",
        "Peneliti": "Gunakan bahasa ilmiah formal. Sertakan nama latin, kelas senyawa, dan terminologi metodologis (IC50, GC-MS, HPLC).",
        "Pelajar": "Gunakan bahasa yang edukatif, mudah dipahami, cocok untuk mahasiswa farmasi/biologi/kedokteran/Kimia SMA.",
        "Umum": "Gunakan bahasa sehari-hari yang sederhana dan mudah dipahami. Hindari jargon teknis. Fokus pada manfaat praktis.",
    }
    persona_instruction = persona_map.get(ai_mode, persona_map["Pelajar"])

    return f"""Anda adalah mesin pembuat kuis adaptif.

TOPIK UTAMA:
{topic}

JUMLAH SOAL DALAM BATCH:
{batch_count}

TOTAL SOAL:
{total_count}

PROFIL PENGGUNA:
{ai_mode}

ADAPTASI PERSONA:
{persona_instruction}

GENERATION NONCE:
{generation_nonce}

Gunakan nonce tersebut hanya untuk memilih kombinasi subtopik, sudut pandang,
jenis pertanyaan, tingkat kesulitan, dan susunan jawaban yang berbeda.
Jangan menuliskan nonce pada output.

BLUEPRINT WAJIB:
{blueprint_json}

KONTEKS REFERENSI:
{context_data}
{file_instruction}

ATURAN:
1. Ikuti blueprint secara ketat untuk setiap nomor soal.
2. Seluruh pertanyaan harus relevan dengan topik dan konteks referensi.
3. Jangan mengulang konsep, kalimat pembuka, skenario, atau opsi jawaban antar nomor soal.
4. Setiap pertanyaan WAJIB memiliki tepat 4 opsi (A, B, C, D) dan hanya ada satu jawaban benar.
5. Buat pengecoh (distractors) yang masuk akal dan berada dalam kategori yang sama.
6. Posisi jawaban benar wajib disesuaikan dengan blueprint target correct_label.
7. Gunakan variasi tingkat kesulitan dan level kognitif sesuai blueprint.
8. Jangan menggunakan informasi di luar konteks untuk klaim yang sangat spesifik.
9. Pembahasan harus menjelaskan jawaban benar dan kekeliruan utama pengecoh.
10. Keluarkan JSON valid sesuai schema QuizResponse secara murni, tanpa pembuka/penutup markdown.
"""

# PARSING, NORMALIZATION, AND DEDUPLICATION LAYER
def normalize_quiz_payload(payload: Any) -> dict[str, Any]:
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except Exception as e:
            logger.warning(f"Failed to parse payload string as JSON: {e}")
            raise QuizParsingError("Payload is not a valid JSON string")

    if not isinstance(payload, dict):
        raise QuizParsingError(f"Expected dict, got {type(payload)}")

    while len(payload) == 1:
        key = list(payload.keys())[0]
        val = payload[key]
        if isinstance(val, dict):
            payload = val
        elif isinstance(val, list):
            if key in ["daftar_soal", "questions", "soal", "list_soal", "items", "data_kuis"]:
                payload = {"daftar_soal": val}
                break
            else:
                break
        else:
            break

    if "arguments" in payload:
        args = payload["arguments"]
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                pass
        if isinstance(args, dict):
            payload = args

    if "render_interactive_quiz" in payload:
        payload = payload["render_interactive_quiz"]

    questions_list = None
    for alt_key in ["daftar_soal", "questions", "soal", "list_soal", "items", "data_kuis", "quiz_questions", "quiz"]:
        if alt_key in payload:
            val = payload[alt_key]
            if isinstance(val, list):
                questions_list = val
                break
            elif isinstance(val, dict) and "questions" in val:
                questions_list = val["questions"]
                break

    if questions_list is None:
        for k, v in payload.items():
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict):
                questions_list = v
                break

    if questions_list is None:
        raise QuizParsingError("Could not find list of questions in payload")

    normalized_questions = []
    for idx, q in enumerate(questions_list):
        if not isinstance(q, dict):
            continue

        id_soal = q.get("id_soal", q.get("id", q.get("question_id", f"Q-{idx+1:02d}")))
        tingkat_kesulitan = q.get("tingkat_kesulitan", q.get("difficulty", q.get("level", "Menengah")))
        pertanyaan = q.get("pertanyaan", q.get("question", q.get("text", q.get("question_text", ""))))

        raw_options = q.get("opsi_jawaban", q.get("options", q.get("choices", q.get("answers", []))))
        normalized_options = []
        if isinstance(raw_options, list):
            for opt_idx, opt in enumerate(raw_options):
                if isinstance(opt, dict):
                    lbl = opt.get("label", opt.get("key", "")).strip().upper()
                    if not lbl:
                        lbl = ["A", "B", "C", "D"][opt_idx % 4]
                    txt = opt.get("text", opt.get("value", ""))
                    normalized_options.append({"label": lbl, "text": txt})
                elif isinstance(opt, str):
                    lbl = ["A", "B", "C", "D"][opt_idx % 4]
                    normalized_options.append({"label": lbl, "text": opt})

        jawaban_benar = q.get("jawaban_benar", q.get("correct_answer", q.get("answer", q.get("correct_label", "A")))).strip().upper()
        
        raw_pembahasan = q.get("pembahasan", q.get("explanation", q.get("rationale", [])))
        if isinstance(raw_pembahasan, list):
            pembahasan = [str(x) for x in raw_pembahasan]
        elif isinstance(raw_pembahasan, str):
            pembahasan = [raw_pembahasan]
        else:
            pembahasan = ["Pembahasan sesuai dengan fakta ilmiah."]

        penjelasan_salah = q.get("penjelasan_salah", q.get("wrong_explanation", q.get("distractor_explanation", "Pilihan lainnya kurang tepat.")))
        
        level_kognitif = q.get("level_kognitif", q.get("cognitive_level", "Memahami"))
        jenis_soal = q.get("jenis_soal", q.get("question_type", "Konsep"))
        subtopik = q.get("subtopik", q.get("subtopic", "Umum"))

        normalized_questions.append({
            "id_soal": id_soal,
            "tingkat_kesulitan": tingkat_kesulitan,
            "pertanyaan": pertanyaan,
            "opsi_jawaban": normalized_options,
            "jawaban_benar": jawaban_benar,
            "pembahasan": pembahasan,
            "penjelasan_salah": penjelasan_salah,
            "level_kognitif": level_kognitif,
            "jenis_soal": jenis_soal,
            "subtopik": subtopik
        })

    normalized_payload = {
        "topik": payload.get("topik", ""),
        "daftar_soal": normalized_questions,
        "analisis_performa": payload.get("analisis_performa", {
            "sorotan": ["Selesai mengerjakan kuis."],
            "area_fokus": ["Lanjutkan mempelajari topik ini."]
        })
    }

    logger.info(
        "Quiz payload normalized: top_keys=%s, question_count=%s",
        list(normalized_payload.keys()),
        len(normalized_payload.get("daftar_soal", [])),
    )
    return normalized_payload

def align_options_to_blueprint(question: dict[str, Any], target_label: str) -> dict[str, Any]:
    options = question.get("opsi_jawaban", [])
    correct_label = question.get("jawaban_benar", "").strip().upper()

    correct_opt = None
    distractors = []
    for opt in options:
        if opt.get("label", "").strip().upper() == correct_label:
            correct_opt = opt.copy()
        else:
            distractors.append(opt.copy())

    if not correct_opt or len(distractors) < 3:
        if not correct_opt and len(options) > 0:
            correct_opt = options[0].copy()
            distractors = options[1:]
        while len(distractors) < 3:
            distractors.append({"label": "", "text": "Pilihan alternatif lainnya."})
        distractors = distractors[:3]

    import secrets
    rng = secrets.SystemRandom()
    rng.shuffle(distractors)

    labels = ["A", "B", "C", "D"]
    assembled_options = []
    dist_idx = 0
    for label in labels:
        if label == target_label:
            opt = correct_opt.copy()
        else:
            opt = distractors[dist_idx].copy()
            dist_idx += 1
        opt["label"] = label
        assembled_options.append(opt)

    question["opsi_jawaban"] = assembled_options
    question["jawaban_benar"] = target_label
    return question

def check_question_relevancy(question: dict[str, Any], topic: str) -> bool:
    topic_lower = topic.lower()
    q_text = question.get("pertanyaan", "").lower()
    explanation = " ".join(question.get("pembahasan", [])) if isinstance(question.get("pembahasan", []), list) else str(question.get("pembahasan", ""))
    explanation = explanation.lower()
    subtopik = question.get("subtopik", "").lower()
    
    topic_words = [w for w in re.split(r"\s+", topic_lower) if len(w) > 3]
    if not topic_words:
        topic_words = [topic_lower]
        
    word_match = any(w in q_text or w in explanation or w in subtopik for w in topic_words)
    return word_match

def detect_duplicate_questions(
    questions: list[dict[str, Any]],
    similarity_threshold: float = 0.88,
) -> list[int]:
    import numpy as np

    def normalize_text(text: str) -> str:
        text = text.lower()
        text = re.sub(r"[^\w\s]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(r"^(q\d+|soal|pertanyaan|\d+)\s*", "", text)
        return text

    normalized_texts = [normalize_text(q.get("pertanyaan", "")) for q in questions]
    duplicates = set()

    seen = {}
    for idx, norm in enumerate(normalized_texts):
        if not norm:
            continue
        if norm in seen:
            duplicates.add(idx)
        else:
            seen[norm] = idx

    non_dup_indices = [i for i in range(len(questions)) if i not in duplicates]
    if len(non_dup_indices) > 1:
        texts_to_embed = [questions[i].get("pertanyaan", "") for i in non_dup_indices]
        try:
            embeddings = [embed_text(t) for t in texts_to_embed]
            embeddings_matrix = np.array(embeddings)
            norms = np.linalg.norm(embeddings_matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            norm_embeddings = embeddings_matrix / norms
            sim_matrix = np.dot(norm_embeddings, norm_embeddings.T)
            
            for i in range(len(non_dup_indices)):
                for j in range(i + 1, len(non_dup_indices)):
                    if sim_matrix[i, j] > similarity_threshold:
                        duplicates.add(non_dup_indices[j])
        except Exception as e:
            logger.warning(f"Semantic deduplication failed: {e}. Falling back to exact matching.")

    return sorted(list(duplicates))

# REPAIR AND FALLBACK LAYERS
async def complete_missing_questions(
    existing_questions: list[dict[str, Any]],
    blueprint: QuizBlueprint,
    required_count: int,
    topic: str,
    context_data: str,
    ai_mode: str,
    model: str,
    file_context: Optional[str] = None,
) -> list[dict[str, Any]]:
    missing_count = required_count - len(existing_questions)
    if missing_count <= 0:
        return existing_questions

    logger.info(f"Quiz repair triggered: current_count={len(existing_questions)}, missing_count={missing_count}")
    
    retries = 2
    for attempt in range(retries):
        missing_count = required_count - len(existing_questions)
        if missing_count <= 0:
            break

        missing_blueprint_items = blueprint.items[len(existing_questions) : required_count]
        missing_blueprint_json = json.dumps([item.model_dump() for item in missing_blueprint_items], indent=2)

        existing_fingerprints = [q.get("pertanyaan", "") for q in existing_questions]
        existing_fingerprints_str = "\n".join([f"- {f}" for f in existing_fingerprints])

        repair_prompt = f"""Anda adalah mesin pembuat kuis adaptif.
Tugas Anda adalah memproduksi persis {missing_count} soal tambahan yang belum ada.

SOAL YANG SUDAH ADA (DILARANG KERAS MEMBUAT SOAL YANG SAMA ATAU SERUPA):
{existing_fingerprints_str}

BLUEPRINT UNTUK SOAL TAMBAHAN:
{missing_blueprint_json}

IKUTI ATURAN SYSTEM SEPENUHNYA DAN KELUARKAN HANYA OBJEK JSON YANG BERISI SOAL TAMBAHAN TERSEBUT."""

        resolved_model = model or settings.LLM_DEFAULT_MODEL
        batch_system_prompt = _build_quiz_system_prompt(
            context_data=context_data,
            batch_count=missing_count,
            total_count=required_count,
            ai_mode=ai_mode,
            topic=topic,
            generation_nonce=secrets.token_hex(16),
            blueprint_json=missing_blueprint_json,
            file_context=file_context,
        )

        try:
            response = _client.chat.completions.create(
                model=resolved_model,
                messages=[
                    {"role": "system", "content": batch_system_prompt},
                    {"role": "user", "content": repair_prompt},
                ],
                temperature=0.8,
                max_tokens=min(12000, 800 + missing_count * 280),
            )
            message = response.choices[0].message
            raw_arguments = _parse_tool_calls(message) or _parse_content_regex(message)
            if raw_arguments:
                normalized = normalize_quiz_payload(raw_arguments)
                new_questions = normalized.get("daftar_soal", [])
                
                for nq in new_questions:
                    if len(existing_questions) >= required_count:
                        break
                    is_dup = False
                    for eq in existing_questions:
                        if nq.get("pertanyaan", "").strip().lower() == eq.get("pertanyaan", "").strip().lower():
                            is_dup = True
                            break
                    if not is_dup:
                        target_bp_item = missing_blueprint_items[len(existing_questions) - len(existing_questions)]
                        nq_aligned = align_options_to_blueprint(nq, target_bp_item.correct_label)
                        nq_aligned["tingkat_kesulitan"] = target_bp_item.tingkat_kesulitan
                        nq_aligned["level_kognitif"] = target_bp_item.level_kognitif
                        nq_aligned["jenis_soal"] = target_bp_item.jenis_soal
                        nq_aligned["subtopik"] = target_bp_item.subtopik
                        existing_questions.append(nq_aligned)
        except Exception as e:
            logger.warning(f"Repair attempt {attempt+1} failed: {e}")

    return existing_questions

def _extract_keywords_from_context(topic: str, context_data: str) -> dict[str, list[str]]:
    words = re.findall(r"\b[A-Za-z]+(?:oid|in|at|ol|olida|asetat|genin|glukosa)\b", context_data, re.IGNORECASE)
    botanical = re.findall(r"\b[A-Z][a-z]+ [a-z]+\b", context_data)
    
    keywords = {
        "compounds": list(set([w.capitalize() for w in words]))[:10],
        "botanical": list(set(botanical))[:5],
    }
    
    if not keywords["compounds"]:
        keywords["compounds"] = ["Kurkumin", "Alkaloid", "Flavonoid", "Saponin", "Tanin", "Terpenoid", "Glikosida"]
    if not keywords["botanical"]:
        keywords["botanical"] = [f"{topic.capitalize()} indica", f"{topic.capitalize()} officinale"]
        
    return keywords

def get_synthetic_patterns(topic: str, keywords: dict[str, list[str]]) -> list[dict[str, Any]]:
    c = keywords["compounds"]
    b = keywords["botanical"]
    t = topic
    
    while len(c) < 5:
        c.append("Senyawa aktif")
    while len(b) < 2:
        b.append("Spesies herbal")

    patterns = [
        {
            "tingkat_kesulitan": "Mudah",
            "pertanyaan": f"Metabolit sekunder golongan apakah yang dominan ditemukan pada {t}?",
            "opsi_jawaban": [c[0], c[1], "Karbohidrat", "Protein"],
            "jawaban_benar": "A",
            "pembahasan": [f"Berdasarkan fitokimia, {t} mengandung {c[0]} sebagai salah satu metabolit sekunder utamanya.", f"{c[0]} berkontribusi besar pada efek terapeutik."],
            "penjelasan_salah": "Karbohidrat dan protein adalah metabolit primer umum, bukan metabolit sekunder bioaktif."
        },
        {
            "tingkat_kesulitan": "Menengah",
            "pertanyaan": f"Metode ekstraksi dingin yang paling sesuai untuk mengisolasi {c[0]} dari {t} adalah...",
            "opsi_jawaban": ["Maserasi", "Sokletasi", "Destilasi uap", "Dekoksi"],
            "jawaban_benar": "A",
            "pembahasan": [f"Maserasi merupakan metode ekstraksi dingin yang mencegah kerusakan termal pada {c[0]}.", "Sokletasi dan dekoksi melibatkan pemanasan tinggi."],
            "penjelasan_salah": "Opsi lain melibatkan pemanasan tinggi yang berisiko mendegradasi senyawa termolabil."
        },
        {
            "tingkat_kesulitan": "Tinggi",
            "pertanyaan": f"Mekanisme farmakologis utama {c[0]} dalam {t} sebagai agen anti-inflamasi adalah...",
            "opsi_jawaban": ["Menghambat enzim COX-2", "Merusak dinding sel bakteri", "Meningkatkan sekresi histamin", "Menekan sintesis antibody"],
            "jawaban_benar": "A",
            "pembahasan": [f"Senyawa {c[0]} bekerja dengan menghambat jalur enzim siklooksigenase (COX-2) secara selektif.", "Hal ini mengurangi produksi prostaglandin."],
            "penjelasan_salah": "Opsi lain tidak menunjukkan aktivitas antiinflamasi yang terarah pada level enzim COX."
        },
        {
            "tingkat_kesulitan": "HOTS",
            "pertanyaan": f"Untuk standardisasi ekstrak {t}, parameter mutu spesifik manakah yang diuji untuk mendeteksi kontaminasi anorganik?",
            "opsi_jawaban": ["Kadar abu tidak larut asam", "Kadar air residu", "Susut pengeringan", "Kadar sari larut etanol"],
            "jawaban_benar": "A",
            "pembahasan": ["Kadar abu tidak larut asam menunjukkan cemaran silikat atau pasir.", "Ini menjamin kemurnian ekstrak herbal dari kotoran tanah."],
            "penjelasan_salah": "Kadar air dan susut pengeringan mengukur kelembaban, sedangkan kadar sari larut mengukur efektivitas ekstraksi."
        },
        {
            "tingkat_kesulitan": "Menengah",
            "pertanyaan": f"Dalam klasifikasi ilmiah, nama taksonomi botani yang tepat untuk tanaman {t} adalah...",
            "opsi_jawaban": [b[0], "Oryza sativa", "Zingiber officinale", "Solanum tuberosum"],
            "jawaban_benar": "A",
            "pembahasan": [f"Spesies tanaman {t} secara ilmiah diklasifikasikan sebagai {b[0]}.", "Nama ini membedakannya dari spesies tanaman pangan umum."],
            "penjelasan_salah": "Opsi B, C, dan D adalah nama ilmiah untuk padi, jahe, dan kentang."
        },
        {
            "tingkat_kesulitan": "Mudah",
            "pertanyaan": f"Senyawa marker yang sering dijadikan acuan mutu pada standardisasi simplisia {t} adalah...",
            "opsi_jawaban": [f"{c[0]} standar", "Klorofil", "Kalsium oksalat", "Selulosa"],
            "jawaban_benar": "A",
            "pembahasan": [f"Standardisasi menggunakan {c[0]} standar memastikan kemurnian dan konsistensi bets ekstrak {t}.", "Klorofil dan selulosa adalah komponen tanaman umum non-terapeutik."],
            "penjelasan_salah": "Klorofil dan kalsium oksalat adalah senyawa penyusun umum tumbuhan yang tidak mencerminkan efek terapi spesifik."
        },
        {
            "tingkat_kesulitan": "Menengah",
            "pertanyaan": f"Untuk menarik senyawa semi-polar dari {t}, pelarut ekstraksi yang paling ideal adalah...",
            "opsi_jawaban": ["Etil asetat", "n-Heksana", "Air suling", "Minyak mineral"],
            "jawaban_benar": "A",
            "pembahasan": ["Etil asetat adalah pelarut semi-polar universal yang ideal.", "n-Heksana menarik senyawa non-polar, sedangkan air menarik senyawa polar."],
            "penjelasan_salah": "n-Heksana dan air berturut-turut terlalu non-polar dan polar, sementara minyak mineral bukan pelarut ekstraksi standar."
        },
        {
            "tingkat_kesulitan": "Tinggi",
            "pertanyaan": f"Reaksi metabolisme Fase II manakah yang dialami oleh flavonoid {t} di dalam hepar?",
            "opsi_jawaban": ["Glukuronidasi", "Oksidasi sitokrom P450", "Hidrolisis esterase", "Reduksi karbonil"],
            "jawaban_benar": "A",
            "pembahasan": ["Senyawa flavonoid mengalami konjugasi dengan asam glukuronat (Glukuronidasi) untuk mempermudah ekskresi.", "Oksidasi adalah reaksi Fase I."],
            "penjelasan_salah": "Opsi B, C, dan D merupakan bagian dari reaksi Fase I, bukan reaksi konjugasi Fase II."
        },
        {
            "tingkat_kesulitan": "HOTS",
            "pertanyaan": f"Gejala efek samping hepatotoksik akibat konsumsi berlebih suplemen {t} ditandai dengan...",
            "opsi_jawaban": ["Peningkatan serum SGOT dan SGPT", "Penurunan urea nitrogen darah", "Penurunan jumlah leukosit", "Rambut rontok akut"],
            "jawaban_benar": "A",
            "pembahasan": ["Kerusakan sel hati (hepatosit) memicu pelepasan transaminase (SGOT/SGPT) ke sirkulasi darah.", "Ini merupakan indikator utama hepatotoksisitas."],
            "penjelasan_salah": "Penurunan urea nitrogen berkaitan dengan ginjal atau malnutrisi, sedangkan opsi lain tidak spesifik untuk organ hepar."
        },
        {
            "tingkat_kesulitan": "Mudah",
            "pertanyaan": f"Pereaksi warna manakah yang positif mendeteksi adanya {c[0]} pada fraksi {t}?",
            "opsi_jawaban": ["Pereaksi khusus golongan", "Pereaksi Fehling", "Pereaksi Ninhidrin", "Larutan kanji"],
            "jawaban_benar": "A",
            "pembahasan": [f"Uji fitokimia spesifik untuk {c[0]} memberikan perubahan warna khas.", "Fehling dan ninhidrin digunakan untuk gula dan asam amino."],
            "penjelasan_salah": "Fehling mendeteksi gula pereduksi, Ninhidrin mendeteksi asam amino, dan kanji menguji amilum."
        },
        {
            "tingkat_kesulitan": "Menengah",
            "pertanyaan": f"Alat instrumen yang paling tepat untuk menguji kadar senyawa volatil pada {t} adalah...",
            "opsi_jawaban": ["GC-MS", "HPLC", "FTIR", "Spektrofotometer UV-Vis"],
            "jawaban_benar": "A",
            "pembahasan": ["Gas Chromatography-Mass Spectrometry (GC-MS) adalah standar emas untuk analisis senyawa atsiri/volatil.", "HPLC digunakan untuk senyawa non-volatil."],
            "penjelasan_salah": "HPLC, FTIR, dan UV-Vis kurang optimal untuk memisahkan campuran senyawa yang mudah menguap."
        },
        {
            "tingkat_kesulitan": "Tinggi",
            "pertanyaan": f"Pemisahan zat aktif {t} berdasarkan perbedaan ukuran partikel/pori dilakukan dengan metode...",
            "opsi_jawaban": ["Kromatografi filtrasi gel", "Kromatografi silika gel", "Kromatografi gas", "Kromatografi ion"],
            "jawaban_benar": "A",
            "pembahasan": ["Kromatografi filtrasi gel (eksklusi ukuran) memisahkan senyawa berdasarkan ukuran molekulnya.", "Silika memisahkan berdasarkan polaritas."],
            "penjelasan_salah": "Kromatografi silika, gas, dan ion memisahkan komponen berdasarkan afinitas polaritas atau muatan ionik."
        },
        {
            "tingkat_kesulitan": "Tinggi",
            "pertanyaan": f"Bagaimanakah mekanisme reduksi radikal DPPH oleh senyawa antioksidan dari {t}?",
            "opsi_jawaban": ["Transfer atom hidrogen atau elektron", "Pemutusan rantai cincin aromatik", "Oksidasi gugus nitro radikal", "Pengendapan kompleks logam"],
            "jawaban_benar": "A",
            "pembahasan": ["Antioksidan mendonorkan atom hidrogen/elektron untuk menstabilkan molekul radikal bebas DPPH.", "Ini mengubah warna ungu menjadi kuning."],
            "penjelasan_salah": "Antioksidan bekerja melalui reduksi donor proton, bukan melalui pemutusan cincin aromatik atau oksidasi."
        },
        {
            "tingkat_kesulitan": "Menengah",
            "pertanyaan": f"Batas maksimum cemaran mikroba patogen pada simplisia {t} diatur ketat untuk mencegah toksin dari...",
            "opsi_jawaban": ["Aflatoksin dari Aspergillus", "Saccharomyces", "Lactobacillus", "Penicillium"],
            "jawaban_benar": "A",
            "pembahasan": ["Aspergillus flavus dapat menghasilkan aflatoksin yang bersifat karsinogenik kuat pada simplisia lembab.", "Saccharomyces adalah ragi fermentasi."],
            "penjelasan_salah": "Saccharomyces dan Lactobacillus bukan mikroba patogen penghasil mikotoksik berbahaya pada penyimpanan herbal."
        },
        {
            "tingkat_kesulitan": "Mudah",
            "pertanyaan": f"Uji flavonoid positif pada ekstrak {t} ditandai dengan terbentuknya warna merah setelah reaksi...",
            "opsi_jawaban": ["Shinoda (Mg + HCl)", "Liebermann-Burchard", "Mayer", "Biuret"],
            "jawaban_benar": "A",
            "pembahasan": ["Uji Shinoda mereduksi flavonoid menggunakan logam Mg dan HCl pekat.", "Uji LB adalah untuk steroid/triterpenoid."],
            "penjelasan_salah": "Liebermann-Burchard mendeteksi steroid/triterpenoid, sedangkan Mayer dan Biuret mendeteksi alkaloid dan protein."
        },
        {
            "tingkat_kesulitan": "Mudah",
            "pertanyaan": f"Bagian tumbuhan (simplisia) dari {t} yang paling sering digunakan dalam pengobatan tradisional adalah...",
            "opsi_jawaban": ["Bagian berkhasiat spesifik", "Dinding selulosa mati", "Klorofil daun kering", "Serat kambium kayu"],
            "jawaban_benar": "A",
            "pembahasan": [f"Bagian berkhasiat spesifik (seperti daun, rimpang, atau kulit batang) mengandung konsentrasi tinggi {c[0]}.", "Serat kambium tidak mengandung zat aktif."],
            "penjelasan_salah": "Selulosa, klorofil, dan serat kambium adalah penyusun sel tanaman non-terapeutik yang tidak memicu efek klinis."
        },
        {
            "tingkat_kesulitan": "Menengah",
            "pertanyaan": f"Uap air digunakan dalam destilasi atsiri {t} dengan tujuan utama untuk...",
            "opsi_jawaban": ["Menurunkan titik didih campuran", "Menghidrolisis metabolit aktif", "Mengoksidasi monoterpen volatil", "Melarutkan selulosa dinding sel"],
            "jawaban_benar": "A",
            "pembahasan": ["Uap air menurunkan suhu penguapan campuran di bawah 100C untuk melindungi zat volatil dari kerusakan.", "Zat volatil tidak boleh dihidrolisis."],
            "penjelasan_salah": "Destilasi uap bertujuan mengevaporasi komponen volatil pada suhu rendah, bukan untuk reaksi hidrolisis atau oksidasi."
        },
        {
            "tingkat_kesulitan": "Tinggi",
            "pertanyaan": f"Penurunan glikogenolisis oleh ekstrak {t} bermanfaat secara klinis bagi penderita...",
            "opsi_jawaban": ["Diabetes melitus (Hipoglikemik)", "Hipertensi esensial", "Hepatitis viral kronis", "Gagal ginjal stadium akhir"],
            "jawaban_benar": "A",
            "pembahasan": [f"Penghambatan pelepasan glukosa hepar oleh {t} menghasilkan efek hipoglikemik.", "Ini sangat penting untuk mengontrol gula darah."],
            "penjelasan_salah": "Efek penurun glukosa darah (hipoglikemik) ditujukan bagi pasien diabetes melitus, bukan hipertensi atau gagal ginjal."
        },
        {
            "tingkat_kesulitan": "HOTS",
            "pertanyaan": f"Nilai parameter LD50 yang tinggi pada uji toksisitas akut {t} menunjukkan...",
            "opsi_jawaban": ["Tingkat keamanan konsumsi tinggi", "Toksisitas akut yang mematikan", "Kelarutan organofilik tinggi", "Kecepatan ekskresi renal tinggi"],
            "jawaban_benar": "A",
            "pembahasan": ["LD50 tinggi menandakan dosis yang diperlukan untuk memicu kematian sangat besar, artinya bahan tersebut relatif aman.", "Bahan toksik memiliki LD50 sangat rendah."],
            "penjelasan_salah": "LD50 yang tinggi berkorelasi dengan rentang dosis aman (margin of safety) yang luas, bukan tingkat racun yang tinggi."
        },
        {
            "tingkat_kesulitan": "Mudah",
            "pertanyaan": f"Bagian non-gula pada struktur senyawa glikosida dari {t} dinamakan...",
            "opsi_jawaban": ["Aglikon (Genin)", "Glikon", "Hemiasetal", "Oligosakarida"],
            "jawaban_benar": "A",
            "pembahasan": ["Bagian non-gula (aglikon) menentukan khasiat farmakologis glikosida.", "Glikon adalah bagian gugus karbohidrat."],
            "penjelasan_salah": "Glikon dan oligosakarida merupakan gugus gula, sedangkan hemiasetal adalah tipe ikatan kimianya."
        }
    ]
    return patterns

def _generate_synthetic_quiz(topic: str, jumlah_soal: int, blueprint: Optional[QuizBlueprint] = None) -> dict[str, Any]:
    logger.warning(
        f"SYNTHETIC FALLBACK TRIGGERED for topic=\'{topic[:50]}\'. "
        f"Generating completely dynamic {jumlah_soal} template quiz array rows."
    )
    
    topic_name = topic if topic else "Tanaman Obat & Kimia Farmasi"
    keywords = _extract_keywords_from_context(topic_name, "")
    patterns = get_synthetic_patterns(topic_name, keywords)
    
    import secrets
    rng = secrets.SystemRandom()
    
    indices = list(range(len(patterns)))
    rng.shuffle(indices)
    
    selected_questions = []
    for i in range(jumlah_soal):
        idx = indices[i % len(patterns)]
        pat = patterns[idx].copy()
        
        target_label = "A"
        if blueprint and i < len(blueprint.items):
            target_label = blueprint.items[i].correct_label
        else:
            target_label = rng.choice(["A", "B", "C", "D"])
            
        raw_options = pat["opsi_jawaban"]
        correct_text = raw_options[0]
        distractors = raw_options[1:]
        rng.shuffle(distractors)
        
        options_dict_list = []
        labels = ["A", "B", "C", "D"]
        dist_idx = 0
        for lbl in labels:
            if lbl == target_label:
                options_dict_list.append({"label": lbl, "text": correct_text})
            else:
                options_dict_list.append({"label": lbl, "text": distractors[dist_idx]})
                dist_idx += 1
                
        q = {
            "id_soal": f"Q-{i+1:02d}",
            "tingkat_kesulitan": pat["tingkat_kesulitan"],
            "pertanyaan": pat["pertanyaan"],
            "opsi_jawaban": options_dict_list,
            "jawaban_benar": target_label,
            "pembahasan": pat["pembahasan"],
            "penjelasan_salah": pat["penjelasan_salah"],
            "level_kognitif": blueprint.items[i].level_kognitif if (blueprint and i < len(blueprint.items)) else "Memahami",
            "jenis_soal": blueprint.items[i].jenis_soal if (blueprint and i < len(blueprint.items)) else pat["tingkat_kesulitan"],
            "subtopik": blueprint.items[i].subtopik if (blueprint and i < len(blueprint.items)) else "Umum"
        }
        selected_questions.append(q)
        
    analisis_performa_data = {
        "sorotan": [
            f"Mampu memetakan ruang lingkup fitokimia dasar komoditas {topic_name}.",
            "Mengidentifikasi pentingnya penanda aktif (marker) dalam standardisasi simplisia."
        ],
        "area_fokus": [
            "Perlu memperdalam jalur metabolisme ADME dan interaksi enzim sitokrom P450.",
            "Disarankan meninjau kembali regulasi standardisasi sediaan ekstrak herbal."
        ]
    }
    
    internal_quiz = {
        "topik": topic_name,
        "daftar_soal": selected_questions,
        "analisis_performa": analisis_performa_data,
        "generation_metadata": {
            "generation_id": blueprint.generation_id if blueprint else str(uuid.uuid4()),
            "source": "synthetic_fallback",
            "fallback_used": True,
            "requested_count": jumlah_soal,
            "generated_count": len(selected_questions),
            "duplicate_regenerations": 0
        }
    }
    
    return _map_to_frontend_payload(internal_quiz, jumlah_soal)

# PARSING AND FRONTEND PAYLOAD COMPATIBILITY LAYER
def _build_tool_schema(jumlah_soal: int) -> list[dict[str, Any]]:
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
    if not hasattr(message, "tool_calls") or not message.tool_calls:
        return None
    try:
        tool_call = message.tool_calls[0]
        args_str = tool_call.function.arguments
        return json.loads(args_str)
    except Exception as e:
        logger.warning(f"Formal tool_calls parsing failed: {e}")
        return None

def _parse_content_regex(message: Any) -> Optional[dict[str, Any]]:
    content = getattr(message, "content", "") or ""
    if not content:
        return None

    content_clean = content.strip()
    if content_clean.startswith("```json") and content_clean.endswith("```"):
        content_clean = content_clean[7:-3].strip()
    elif content_clean.startswith("```") and content_clean.endswith("```"):
        content_clean = content_clean[3:-3].strip()

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
            "tingkat_kesulitan": item.get("tingkat_kesulitan", "Menengah"),
            "level_kognitif": item.get("level_kognitif", "Memahami"),
            "jenis_soal": item.get("jenis_soal", "Konsep"),
            "subtopik": item.get("subtopik", "Umum")
        })
    return {
        "topik": validated_quiz_dict.get("topik", ""),
        "daftar_soal": source_questions,
        "analisis_performa": validated_quiz_dict.get("analisis_performa", {}),
        "questions": mapped_questions,
        "generation_metadata": validated_quiz_dict.get("generation_metadata", {})
    }

# MAIN INTERACTIVE QUIZ TOOL ENDPOINT (ASYNC DEFINITION)
async def generate_interactive_quiz_tool(
    topic: str,
    jumlah_soal: int = 3,
    ai_mode: str = "Pelajar",
    file_context: Optional[str] = None,
    model: Optional[str] = None,
) -> dict[str, Any]:
    logger.info(
        f"Quiz pipeline started: raw_topic=\'{topic[:60]}\', "
        f"jumlah_soal={jumlah_soal}, mode={ai_mode}"
    )

    # 1. NLP Preprocessing & Clamp
    extracted_jumlah = _extract_jumlah_soal(topic, default=jumlah_soal)
    cleaned_topic = _clean_topic(topic)
    jumlah_soal = extracted_jumlah

    logger.info(
        f"After NLP preprocessing: topic=\'{cleaned_topic}\', "
        f"jumlah_soal={jumlah_soal}"
    )

    # 2. Adaptive Hybrid Retrieval
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

    _MAX_CONTEXT_CHARS = 4000
    if len(context_data) > _MAX_CONTEXT_CHARS:
        context_data = context_data[:_MAX_CONTEXT_CHARS]

    # Create Quiz Blueprint
    domain = _detect_broad_domain(cleaned_topic)
    blueprint = create_quiz_blueprint(cleaned_topic, jumlah_soal, domain)
    logger.info(f"Quiz Blueprint created: generation_id={blueprint.generation_id}")

    # 3. Batching Generation
    batch_size_limit = 10
    num_batches = (jumlah_soal + batch_size_limit - 1) // batch_size_limit
    batch_sizes = [jumlah_soal // num_batches] * num_batches
    for idx in range(jumlah_soal % num_batches):
        batch_sizes[idx] += 1

    all_generated_questions = []
    duplicate_regenerations = 0
    resolved_model = model or settings.LLM_DEFAULT_MODEL

    start_idx = 0
    for b_idx, b_size in enumerate(batch_sizes):
        logger.info(f"Generating batch {b_idx+1}/{num_batches} of size {b_size}")
        batch_blueprint_items = blueprint.items[start_idx : start_idx + b_size]
        batch_blueprint_json = json.dumps([item.model_dump() for item in batch_blueprint_items], indent=2)

        generation_nonce = secrets.token_hex(16)
        system_prompt = _build_quiz_system_prompt(
            context_data=context_data,
            batch_count=b_size,
            total_count=jumlah_soal,
            ai_mode=ai_mode,
            topic=cleaned_topic,
            generation_nonce=generation_nonce,
            blueprint_json=batch_blueprint_json,
            file_context=file_context,
        )

        tools = _build_tool_schema(b_size)
        raw_arguments = None
        
        # Tier 1: Forced Tool calling
        try:
            logger.info("Attempting forced tool calling...")
            response = _client.chat.completions.create(
                model=resolved_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Hasilkan kuis untuk batch ini: {cleaned_topic}."},
                ],
                temperature=0.8,
                max_tokens=min(12000, 800 + b_size * 280),
                tools=tools,
                tool_choice={
                    "type": "function",
                    "function": {"name": "render_interactive_quiz"}
                }
            )
            raw_arguments = _parse_tool_calls(response.choices[0].message)
        except Exception as e:
            logger.warning(f"Tier 1: Forced tool calling failed: {e}. Trying Tier 2...")

        # Tier 2: JSON Schema response format
        if not raw_arguments:
            try:
                logger.info("Attempting structured JSON response format...")
                response = _client.chat.completions.create(
                    model=resolved_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Hasilkan objek JSON kuis murni berisi {b_size} soal untuk: {cleaned_topic}."},
                    ],
                    temperature=0.8,
                    max_tokens=min(12000, 800 + b_size * 280),
                    response_format={"type": "json_object"}
                )
                raw_arguments = _parse_content_regex(response.choices[0].message)
            except Exception as e:
                logger.warning(f"Tier 2: JSON response format failed: {e}. Trying Tier 3...")

        # Tier 3: Standard completions with structured JSON prompt
        if not raw_arguments:
            try:
                logger.info("Attempting standard structured JSON prompt...")
                response = _client.chat.completions.create(
                    model=resolved_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Hasilkan objek JSON kuis murni berisi {b_size} soal untuk: {cleaned_topic}."},
                    ],
                    temperature=0.8,
                    max_tokens=min(12000, 800 + b_size * 280)
                )
                raw_arguments = _parse_content_regex(response.choices[0].message)
            except Exception as e:
                logger.error(f"Tier 3: Standard completion failed: {e}.")

        if raw_arguments:
            try:
                normalized = normalize_quiz_payload(raw_arguments)
                batch_questions = normalized.get("daftar_soal", [])
                
                valid_batch_questions = []
                for q_offset, q in enumerate(batch_questions):
                    if q_offset < len(batch_blueprint_items):
                        target_bp = batch_blueprint_items[q_offset]
                        q = align_options_to_blueprint(q, target_bp.correct_label)
                        q["tingkat_kesulitan"] = target_bp.tingkat_kesulitan
                        q["level_kognitif"] = target_bp.level_kognitif
                        q["jenis_soal"] = target_bp.jenis_soal
                        q["subtopik"] = target_bp.subtopik

                    if check_question_relevancy(q, cleaned_topic):
                        valid_batch_questions.append(q)
                    else:
                        logger.warning(f"Question was rejected due to low relevancy: {q.get('pertanyaan')}")
                        
                all_generated_questions.extend(valid_batch_questions)
            except Exception as exc:
                logger.exception("Batch parsing or validation failed: %s", exc)

        start_idx += b_size

    # 4. Partial Questions Repair
    if len(all_generated_questions) < jumlah_soal:
        logger.info(f"Partial generation failed to produce enough questions. Found {len(all_generated_questions)}/{jumlah_soal}. Repairing...")
        try:
            all_generated_questions = await complete_missing_questions(
                existing_questions=all_generated_questions,
                blueprint=blueprint,
                required_count=jumlah_soal,
                topic=cleaned_topic,
                context_data=context_data,
                ai_mode=ai_mode,
                model=resolved_model,
                file_context=file_context,
            )
        except Exception as e:
            logger.error(f"Partial repair failed: {e}", exc_info=True)

    # 5. Deduplication and Regeneration
    if len(all_generated_questions) > 0:
        dup_indices = detect_duplicate_questions(all_generated_questions)
        if dup_indices:
            logger.info(f"Detected duplicate indices: {dup_indices}. Regenerating duplicates...")
            duplicate_regenerations += len(dup_indices)
            
            cleaned_questions = [q for idx, q in enumerate(all_generated_questions) if idx not in dup_indices]
            all_generated_questions = cleaned_questions
            
            if len(all_generated_questions) < jumlah_soal:
                try:
                    all_generated_questions = await complete_missing_questions(
                        existing_questions=all_generated_questions,
                        blueprint=blueprint,
                        required_count=jumlah_soal,
                        topic=cleaned_topic,
                        context_data=context_data,
                        ai_mode=ai_mode,
                        model=resolved_model,
                        file_context=file_context,
                    )
                except Exception as e:
                    logger.error(f"Duplicate repair failed: {e}", exc_info=True)

    # If all API pipelines failed completely and we have 0 questions, trigger synthetic fallback
    if len(all_generated_questions) == 0:
        logger.warning("All LLM generation pipelines and repairs failed. Triggering synthetic fallback.")
        return _generate_synthetic_quiz(cleaned_topic, jumlah_soal, blueprint)

    all_generated_questions = all_generated_questions[:jumlah_soal]
    
    if len(all_generated_questions) < jumlah_soal:
        logger.warning(f"Could not reach {jumlah_soal} questions even after repairs. Padding with fallback questions.")
        fallback_quiz = _generate_synthetic_quiz(cleaned_topic, jumlah_soal, blueprint)
        fallback_questions = fallback_quiz.get("daftar_soal", [])
        while len(all_generated_questions) < jumlah_soal:
            pad_q = fallback_questions[len(all_generated_questions) % len(fallback_questions)].copy()
            all_generated_questions.append(pad_q)

    # Ensure sequential Q-01 to Q-N
    for i, q in enumerate(all_generated_questions):
        q["id_soal"] = f"Q-{i+1:02d}"

    session_key = f"{cleaned_topic}_{ai_mode}"
    if session_key not in SESSION_QUIZ_HISTORY:
        SESSION_QUIZ_HISTORY[session_key] = []
    for q in all_generated_questions:
        SESSION_QUIZ_HISTORY[session_key].append(q.get("pertanyaan", ""))
    SESSION_QUIZ_HISTORY[session_key] = SESSION_QUIZ_HISTORY[session_key][-100:]

    final_quiz = {
        "topik": cleaned_topic,
        "daftar_soal": all_generated_questions,
        "analisis_performa": {
            "sorotan": [
                f"Memetakan ruang lingkup kuis untuk materi {cleaned_topic}.",
                "Menguasai pemahaman dasar mengenai topik yang ditanyakan."
            ],
            "area_fokus": [
                "Perlu memperdalam aspek klinis, laboratorium, atau teoretis tingkat lanjut.",
                "Tinjau kembali literatur untuk memperkaya basis teori."
            ]
        },
        "generation_metadata": {
            "generation_id": blueprint.generation_id,
            "source": "llm",
            "fallback_used": False,
            "requested_count": jumlah_soal,
            "generated_count": len(all_generated_questions),
            "duplicate_regenerations": duplicate_regenerations
        }
    }

    logger.info(f"Quiz generated successfully: count={len(all_generated_questions)}")
    return _map_to_frontend_payload(final_quiz, jumlah_soal)

# Startup logging block
def log_startup():
    try:
        hasher = hashlib.md5()
        with open(__file__, "rb") as f:
            hasher.update(f.read())
        md5_hash = hasher.hexdigest()
    except Exception:
        md5_hash = "unknown"

    logger.info("=" * 60)
    logger.info(f"Quiz Generator Loaded:")
    logger.info(f"  Version: {QUIZ_GENERATOR_VERSION}")
    logger.info(f"  Absolute Path: {os.path.abspath(__file__)}")
    logger.info(f"  MAX_QUIZ_QUESTIONS: {MAX_QUIZ_QUESTIONS}")
    logger.info(f"  Active Model: {settings.LLM_DEFAULT_MODEL}")
    logger.info(f"  File MD5 Hash: {md5_hash}")
    logger.info("=" * 60)

log_startup()
