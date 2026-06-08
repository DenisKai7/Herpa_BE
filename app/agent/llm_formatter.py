"""
LLM Formatter - Generates AI responses using HuggingFace Inference API.
"""

import logging
from typing import Generator, Optional, Any

from huggingface_hub import InferenceClient

from app.core.config import settings
from app.core.dependencies import Persona, ModelTier, PERSONA_ALIASES, resolve_model
from app.agent.resolver import resolve_plant_identity

logger = logging.getLogger(__name__)

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# LLM CLIENT (Singleton) - HuggingFace Inference API
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
_client = InferenceClient(
    provider="auto",
    api_key=settings.HF_API_TOKEN,
)

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# MODEL REGISTRY
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
MODEL_REGISTRY = {
    ModelTier.FAST: {
        "model_id": settings.MODEL_FAST,
        "label": "Fast Medium",
        "provider": "hf_router",
        "max_tokens": settings.FAST_MAX_TOKENS,
        "temperature": settings.FAST_TEMPERATURE,
        "retrieval_limit": 5,
        "graph_limit": 4,
        "self_review": False,
    },
    ModelTier.THINKING: {
        "model_id": settings.MODEL_THINKING,
        "label": "Thinking High",
        "provider": "hf_router",
        "max_tokens": settings.THINKING_MAX_TOKENS,
        "temperature": settings.THINKING_TEMPERATURE,
        "retrieval_limit": 10,
        "graph_limit": 8,
        "self_review": True,
    },
}

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# BASE SYSTEM PROMPT
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
BASE_SYSTEM_PROMPT = """
Anda adalah MedBot AI, sistem Agentic AI untuk edukasi,
riset, farmasi, farmakologi, farmakognosi, dan tanaman herbal.

Jawab pertanyaan pengguna saat ini secara relevan.
Gunakan data retrieval yang benar-benar berkaitan dengan pertanyaan.
Jangan mengarang data, nama latin, senyawa, mekanisme,
dosis, referensi, atau tingkat bukti.

Prioritas sumber:
1. dokumen yang diunggah pengguna;
2. database pendidikan;
3. database tanaman;
4. knowledge graph;
5. corpus tervalidasi;
6. pengetahuan model sebagai fallback dengan keterbatasan eksplisit.

Bedakan penggunaan tradisional dengan bukti ilmiah.
Jangan menjanjikan kesembuhan.
Jangan memberikan diagnosis final.
Untuk risiko serius atau red flags, sarankan pemeriksaan profesional.
Gunakan bahasa Indonesia sesuai persona.
"""

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# PERSONA PROMPTS
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
PERSONA_PROMPTS = {
    Persona.UMUM: """PERSONA: UMUM

Anda menjawab untuk masyarakat awam.

Gunakan bahasa Indonesia yang sederhana, jelas, dan ramah.
Jelaskan istilah medis atau kimia menggunakan bahasa sehari-hari.
Jawab inti pertanyaan terlebih dahulu.
Gunakan contoh praktis jika sesuai.

Untuk cara pengolahan herbal:
- berikan cara kebersihan dan pengolahan yang wajar;
- jangan menjanjikan kesembuhan;
- jangan memberikan dosis medis presisi tanpa sumber;
- jelaskan risiko penggunaan berlebihan;
- jelaskan kelompok yang perlu berhati-hati.

Bedakan:
- penggunaan tradisional;
- hasil penelitian awal;
- bukti klinis.

Tanaman herbal adalah pendamping, bukan pengganti pemeriksaan
atau pengobatan medis.""",

    Persona.PELAJAR: """PERSONA: PELAJAR

Anda menjawab untuk pelajar SMA dan mahasiswa.

Gunakan bahasa ilmiah tingkat pendidikan, tetapi tetap jelas.
Jelaskan istilah teknis saat pertama kali digunakan.
Mulai dari konsep dasar, kemudian lanjutkan ke konsep yang lebih kompleks.
Gunakan contoh, analogi, atau perbandingan jika membantu.

Jelaskan:
- identitas botani;
- bagian tanaman;
- kelompok metabolit sekunder;
- fungsi senyawa;
- mekanisme farmakologi dasar;
- keterkaitan dengan biologi dan kimia.

Jangan mengubah jawaban menjadi konsultasi pengobatan individual.
Fokus pada pembelajaran dan pemahaman konsep.""",

    Persona.PENELITI: """PERSONA: PENELITI

Anda menjawab untuk peneliti farmasi, farmakologi,
fitokimia, kimia bahan alam, dan biologi molekuler.

Gunakan bahasa akademik, analitis, dan sistematis.

Bahas sesuai relevansi:
- nomenklatur botani;
- organ tanaman;
- metabolit sekunder;
- marker compound;
- target molekuler;
- jalur pensinyalan;
- hubungan struktur-aktivitas;
- farmakokinetik;
- bioavailabilitas;
- metode ekstraksi;
- metode analitik;
- model eksperimental;
- tingkat bukti;
- keterbatasan;
- research gap.

Bedakan dengan tegas:
- fakta yang didukung konteks;
- inferensi ilmiah;
- hipotesis;
- klaim yang belum terbukti.

Jangan membuat DOI, jurnal, sitasi, angka konsentrasi,
atau hasil penelitian yang tidak terdapat pada sumber retrieval.""",

    Persona.TENAGA_MEDIS: """PERSONA: TENAGA MEDIS

Anda menjawab untuk tenaga medis, farmasis,
dan pengguna bidang farmakognosi.

Gunakan terminologi medis, farmasi, dan farmakologi yang tepat.
Jawaban harus profesional, ringkas, dan berorientasi klinis.

Bahas bila relevan:
- kandungan atau marker;
- farmakodinamik;
- farmakokinetik;
- efikasi;
- kualitas bukti;
- kontraindikasi;
- interaksi obat-herbal;
- efek samping;
- kehamilan;
- menyusui;
- anak;
- lansia;
- gangguan hati;
- gangguan ginjal;
- monitoring;
- red flags.

Jangan memberikan diagnosis final.
Jangan menggantikan keputusan klinis.
Jangan memberikan dosis individual tanpa konteks pasien dan sumber kuat."""
}

STRUCTURE_INSTRUCTIONS = {
    Persona.UMUM: "\n\nUntuk pertanyaan yang relevan, susun jawaban dengan struktur berikut:\n1. Ringkasan\n2. Kandungan utama\n3. Manfaat yang dikenal\n4. Cara pengolahan sederhana\n5. Cara penggunaan yang lebih aman\n6. Siapa yang perlu berhati-hati\n7. Kapan harus ke dokter",
    Persona.PELAJAR: "\n\nUntuk pertanyaan yang relevan, susun jawaban dengan struktur berikut:\n1. Konsep utama\n2. Identitas tanaman\n3. Senyawa aktif\n4. Klasifikasi senyawa\n5. Mekanisme dasar\n6. Contoh penerapan\n7. Ringkasan belajar\n8. Istilah penting",
    Persona.PENELITI: "\n\nUntuk pertanyaan yang relevan, susun jawaban dengan struktur berikut:\n1. Identitas Taksonomi\n2. Bagian Tanaman dan Simplisia\n3. Profil Fitokimia\n4. Marker Compound\n5. Target Molekuler\n6. Mekanisme Biologis\n7. Farmakokinetik dan Bioavailabilitas\n8. Metode Ekstraksi dan Analisis\n9. Evidence Mapping\n10. Keterbatasan Bukti\n11. Research Gap\n12. Hipotesis atau Saran Penelitian",
    Persona.TENAGA_MEDIS: "\n\nUntuk pertanyaan yang relevan, susun jawaban dengan struktur berikut:\n1. Ringkasan Klinis\n2. Kandungan atau Marker Utama\n3. Farmakodinamik\n4. Farmakokinetik\n5. Indikasi Tradisional dan Bukti Klinis\n6. Kontraindikasi\n7. Interaksi Obat-Herbal\n8. Efek Samping\n9. Populasi Khusus\n10. Monitoring\n11. Red Flags\n12. Kualitas Bukti"
}

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# MODEL TIER PROMPTS
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
MODEL_TIER_PROMPTS = {
    ModelTier.FAST: """MODE: FAST MEDIUM

Jawab langsung dan efisien.
Prioritaskan informasi paling penting.
Gunakan maksimal beberapa bagian utama.
Jangan memperpanjang jawaban jika pertanyaan sederhana.
Tetap lakukan pemeriksaan relevansi dan keselamatan.""",

    ModelTier.THINKING: """MODE: THINKING HIGH

Lakukan analisis internal sebelum memberikan jawaban final.
Periksa:
- relevansi konteks;
- konsistensi nama ilmiah;
- validitas senyawa aktif;
- kekuatan bukti;
- keamanan;
- kemungkinan klaim berlebihan;
- kesesuaian bahasa dengan persona.

Jangan tampilkan proses berpikir internal.
Berikan hanya jawaban final yang telah diperiksa."""
}

INTENT_INSTRUCTIONS: dict[str, str] = {
    "konsultasi": (
        "Berikan rekomendasi tanaman obat/herbal berdasarkan gejala. "
        "Sertakan cara penggunaan, dosis umum, dan peringatan. "
        "SELALU ingatkan untuk berkonsultasi dengan tenaga medis profesional."
    ),
    "ensiklopedia": (
        "Berikan informasi ensiklopedis yang lengkap: deskripsi, klasifikasi, "
        "habitat, kandungan fitokimia, dan khasiat. Format sebagai entri "
        "referensi yang terstruktur."
    ),
    "edukasi": (
        "Jelaskan materi edukasi secara bertahap dan terstruktur. "
        "Gunakan format pembelajaran: Definisi -> Penjelasan -> "
        "Contoh -> Kesimpulan."
    ),
}

def normalize_persona(ai_mode: str) -> Persona:
    """Mengonversi input string persona ke enum Persona."""
    val = str(ai_mode).lower().strip()
    return PERSONA_ALIASES.get(val, Persona.UMUM)

def _build_system_prompt(
    query: str,
    context: str,
    persona: Persona,
    model_tier: ModelTier,
    intent: str,
    file_context: Optional[str] = None,
) -> str:
    """Membangun system prompt terstruktur untuk LLM."""
    persona_prompt = PERSONA_PROMPTS.get(persona, PERSONA_PROMPTS[Persona.UMUM])
    structure_instruction = STRUCTURE_INSTRUCTIONS.get(persona, "")
    tier_prompt = MODEL_TIER_PROMPTS.get(model_tier, MODEL_TIER_PROMPTS[ModelTier.FAST])
    intent_instruction = INTENT_INSTRUCTIONS.get(
        intent, "Jawab pertanyaan berdasarkan data yang tersedia."
    )

    system_message = BASE_SYSTEM_PROMPT
    system_message += f"\nâ•â•â• {persona_prompt} â•â•â•"
    system_message += f"\n\nâ•â•â• {tier_prompt} â•â•â•"

    # Plant identity resolution & verification injection
    identity = resolve_plant_identity(query)
    if identity:
        synonyms_str = ", ".join(identity.synonyms) if identity.synonyms else "Tidak tersedia"
        system_message += f"""

â•â•â• VALIDASI IDENTITAS TANAMAN (CANONICAL FACTS) â•â•â•
Nama Lokal: {identity.local_name}
Nama Ilmiah: {identity.scientific_name or 'Tidak tersedia'}
Famili: {identity.family or 'Tidak tersedia'}
Sinonim: {synonyms_str}

Peraturan Mutlak: Anda WAJIB menggunakan identitas ilmiah di atas jika membahas tentang tanaman tersebut. Dilarang keras menukar nama ilmiah atau mencampurkannya dengan spesies lain (misal: dilarang keras mencampurkan Curcuma xanthorrhiza dengan Curcuma zedoaria atau Curcuma longa)."""

    system_message += f"\n\nâ•â•â• INSTRUKSI INTENT: {intent.upper()} â•â•â•\n{intent_instruction}"
    system_message += structure_instruction

    if file_context and file_context.strip():
        file_context_buffer = file_context[:12000]
        system_message += f"\n\n{file_context_buffer.strip()}\n"
        system_message += (
            "\nATURAN KERAS KONTEKS ATTACHMENT:\n"
            "- Jawab berdasarkan ATTACHMENT EVIDENCE dan Neo4j evidence di atas.\n"
            "- Prioritas konteks: attachment user saat ini, chunk attachment chat saat ini, Neo4j verification, Supabase RAG, corpus internal, lalu pengetahuan model.\n"
            "- Jika verification_status bukan verified, jangan menyatakan identitas senyawa/tanaman sebagai pasti. Gunakan istilah seperti 'kandidat', 'paling konsisten', atau 'belum cukup bukti'.\n"
            "- Jika gambar berupa skeletal chemical structure, GOT-OCR2 hanya membaca label/angka/teks visual. Jangan mengarang SMILES, InChI, tanaman asal, khasiat klinis, atau diagnosis dari bentuk struktur saja.\n"
            "- Jika bukti attachment dan database bertentangan, jelaskan konflik, turunkan confidence, dan sebutkan data tambahan yang dibutuhkan.\n"
            "- Jika OCR/Neo4j gagal, tetap jawab bagian yang terbaca sebagai hasil sementara dengan keterbatasan eksplisit.\n"
            "- Semua output wajib Bahasa Indonesia dan sesuai persona.\n"
        )
    system_message += f"""

â•â•â• DATA DATABASE MULAI â•â•â•
{context}
â•â•â• DATA DATABASE SELESAI â•â•â•"""

    return system_message

def generate_strict_response(
    query: str,
    context: str,
    ai_mode: str,
    intent: str,
    file_context: Optional[str] = None,
    model: Optional[str] = None,
    model_tier: Optional[str] = None,
) -> str:
    """Generate respons AI dengan model routing dan format terstruktur."""
    persona = normalize_persona(ai_mode)

    # Resolve model tier
    resolved_tier = ModelTier.FAST
    if model_tier:
        resolved_tier = ModelTier.THINKING if str(model_tier).lower() == "thinking" else ModelTier.FAST
    elif model:
        resolved_tier = ModelTier.THINKING if model == settings.MODEL_THINKING else ModelTier.FAST

    route = resolve_model(resolved_tier, model)
    registry_conf = MODEL_REGISTRY[route.model_tier]

    system_prompt = _build_system_prompt(
        query=query,
        context=context,
        persona=persona,
        model_tier=route.model_tier,
        intent=intent,
        file_context=file_context
    )

    try:
        res = _client.chat.completions.create(
            model=route.used_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ],
            temperature=registry_conf["temperature"],
            max_tokens=registry_conf["max_tokens"],
        )

        content = res.choices[0].message.content
        if content is None:
            logger.warning("LLM returned None content, returning fallback message.")
            return "Maaf, tidak ada respons yang dihasilkan. Silakan coba lagi."

        return content

    except Exception as e:
        logger.error(f"HuggingFace LLM generation error: {e}", exc_info=True)
        return (
            f"Maaf, terjadi kesalahan saat memproses permintaan Anda. "
            f"Silakan coba lagi nanti. (Error: {type(e).__name__})"
        )

def generate_streaming_response(
    query: str,
    context: str,
    ai_mode: str,
    intent: str,
    file_context: Optional[str] = None,
    model: Optional[str] = None,
    model_tier: Optional[str] = None,
) -> Generator[str, None, None]:
    """Generator streaming token dari LLM."""
    persona = normalize_persona(ai_mode)

    # Resolve model tier
    resolved_tier = ModelTier.FAST
    if model_tier:
        resolved_tier = ModelTier.THINKING if str(model_tier).lower() == "thinking" else ModelTier.FAST
    elif model:
        resolved_tier = ModelTier.THINKING if model == settings.MODEL_THINKING else ModelTier.FAST

    route = resolve_model(resolved_tier, model)
    registry_conf = MODEL_REGISTRY[route.model_tier]

    system_prompt = _build_system_prompt(
        query=query,
        context=context,
        persona=persona,
        model_tier=route.model_tier,
        intent=intent,
        file_context=file_context
    )

    try:
        stream = _client.chat.completions.create(
            model=route.used_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ],
            temperature=registry_conf["temperature"],
            max_tokens=registry_conf["max_tokens"],
            stream=True,
        )
        for chunk in stream:
            delta_content = chunk.choices[0].delta.content
            if delta_content:
                yield delta_content

    except Exception as e:
        logger.error(f"HuggingFace LLM streaming error: {e}", exc_info=True)
        yield f"[Error: {type(e).__name__}]"

def generate_structured_recommendation(
    query: str,
    context: str,
) -> str:
    """Menghasilkan rekomendasi herbal dalam format JSON list yang valid."""
    system_prompt = f"""Anda adalah Sistem AI Asisten Klinis dan Pakar Fitokimia Medis Tingkat Tinggi.
Tugas Anda adalah memberikan rekomendasi tanaman obat/herbal berdasarkan keluhan/gejala pasien.

â•â•â• DATA DATABASE KONSULTASI â•â•â•
{context}
â•â•â• AKHIR DATA DATABASE â•â•â•

â•â•â• PETUNJUK FORMAT JAWABAN â•â•â•
Anda WAJIB menghasilkan output dalam format JSON array yang VALID. Setiap objek dalam array merepresentasikan tanaman obat pendukung dan harus memiliki struktur berikut secara persis:
{{
  "tanaman": "Nama populer tanaman obat dalam Bahasa Indonesia",
  "nama_latin": "Nama ilmiah botani resmi (contoh: Curcuma xanthorrhiza)",
  "deskripsi_singkat": "Penjelasan klinis/farmakologis mengapa tanaman ini cocok menyembuhkan gejala tersebut berdasarkan data database",
  "pengolahan_rumahan": "Langkah-langkah terperinci cara membuat ramuan mandiri di rumah (misal: merebus, menyeduh)",
  "aturan_pakai": "Dosis aman, frekuensi konsumsi per hari, dan waktu konsumsi terbaik (misal: sebelum/setelah makan)",
  "peringatan": "Kontraindikasi klinis, potensi efek samping, tingkat toksisitas, dan risiko jika berinteraksi dengan obat kimia/medis"
}}

â•â•â• INSTRUKSI MUTLAK â•â•â•
1. HANYA gunakan informasi yang sahih dari data database di atas. Jangan mengarang informasi.
2. Pastikan output hanya berupa JSON array. Dilarang menyertakan teks pembuka (seperti "Berikut adalah...", "Tentu saja...") atau teks penutup. Langsung berikan JSON array dimulai dengan [ dan diakhiri dengan ].
3. Seluruh isi teks di dalam nilai properti harus ditulis menggunakan Bahasa Indonesia yang profesional dan akademis.
4. Jika tidak ada data tanaman yang cocok di dalam database, kembalikan array kosong: []"""

    try:
        # Paksa menggunakan MODEL_THINKING (Qwen/Qwen2.5-7B-Instruct) karena 14B tidak tersedia
        res = _client.chat.completions.create(
            model=settings.MODEL_THINKING,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Berikan rekomendasi untuk keluhan/gejala: {query}"},
            ],
            temperature=0.0,
            max_tokens=2500,
        )

        content = res.choices[0].message.content
        if not content:
            return "[]"
        return content.strip()

    except Exception as e:
        logger.error(f"Structured recommendation generation failed: {e}", exc_info=True)
        return "[]"

