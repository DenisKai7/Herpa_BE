"""
NLU Intent Router - Klasifikasi intent query pengguna.

Menggunakan pipeline Scikit-Learn (TF-IDF + LinearSVC) yang di-train dari
train_intent_model.py. Jika model .pkl tidak tersedia, fallback ke
rule-based keyword matching.

Intent yang didukung:
- konsultasi: rekomendasi tanaman obat berdasarkan gejala.
- ensiklopedia: pencarian informasi detail tanaman/senyawa.
- edukasi: materi edukasi kimia/farmasi/biologi.
- generate_quiz: pembuatan kuis interaktif.
"""

import logging
import os
from typing import Optional

import joblib
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════
# MODEL LOADING (Lazy Singleton)
# ═══════════════════════════════════════════
_intent_classifier: Optional[Pipeline] = None
_model_loaded: bool = False

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "intent_model.pkl")

# Intent yang valid sesuai pipeline arsitektur
VALID_INTENTS = frozenset({"konsultasi", "ensiklopedia", "edukasi", "generate_quiz"})

# Keyword fallback mapping (digunakan jika model .pkl tidak tersedia)
_FALLBACK_KEYWORDS: dict[str, list[str]] = {
    "generate_quiz": [
        "kuis", "quiz", "soal", "latihan", "ujian", "tes ", "test ",
        "uji pemahaman", "uji pengetahuan", "buat soal", "generate pertanyaan",
    ],
    "konsultasi": [
        "sakit", "obat", "khasiat", "gejala", "rekomendasi", "keluhan",
        "nyeri", "demam", "batuk", "mual", "herbal untuk", "ramuan",
        "interaksi", "kontraindikasi", "dosis", "efek samping",
    ],
    "edukasi": [
        "jelaskan", "reaksi", "kimia", "stoikiometri", "hukum", "konsep",
        "cara menghitung", "apa itu", "apa bedanya", "perbedaan",
        "bagaimana cara", "tolong jelaskan", "terangkan", "biosintesis",
    ],
}


def _load_model() -> Optional[Pipeline]:
    """
    Memuat model intent classifier dari file .pkl.

    Returns:
        Pipeline sklearn jika berhasil dimuat, None jika file tidak ditemukan.
    """
    global _intent_classifier, _model_loaded

    if _model_loaded:
        return _intent_classifier

    try:
        if os.path.exists(_MODEL_PATH):
            _intent_classifier = joblib.load(_MODEL_PATH)
            logger.info(f"Intent model loaded successfully from: {_MODEL_PATH}")
        else:
            logger.warning(
                f"Intent model not found at {_MODEL_PATH}. "
                "Falling back to keyword-based classification. "
                "Run train_intent_model.py to generate the model."
            )
            _intent_classifier = None
    except Exception as e:
        logger.error(f"Failed to load intent model: {e}", exc_info=True)
        _intent_classifier = None

    _model_loaded = True
    return _intent_classifier


def _keyword_fallback(query: str) -> str:
    """
    Fallback intent classification berbasis keyword matching.

    Digunakan ketika model .pkl tidak tersedia. Mencocokkan query
    dengan daftar keyword per-intent secara berurutan (prioritas).

    Args:
        query: Teks query pengguna yang sudah di-lowercase.

    Returns:
        Intent string: salah satu dari VALID_INTENTS.
    """
    query_lower = query.lower()

    for intent, keywords in _FALLBACK_KEYWORDS.items():
        if any(kw in query_lower for kw in keywords):
            logger.debug(f"Keyword fallback matched intent: '{intent}'")
            return intent

    return "ensiklopedia"


def classify_intent(query: str) -> str:
    """
    Mengklasifikasikan intent dari query pengguna.

    Pipeline:
    1. Jika model SVM tersedia -> gunakan model prediksi.
    2. Jika model tidak tersedia -> gunakan keyword fallback.
    3. Validasi hasil terhadap VALID_INTENTS.

    Args:
        query: Teks query dari pengguna (bahasa Indonesia).

    Returns:
        Intent string: 'konsultasi', 'ensiklopedia', 'edukasi', atau 'generate_quiz'.
    """
    if not query or not query.strip():
        logger.warning("Empty query received for intent classification.")
        return "ensiklopedia"

    model = _load_model()

    if model is not None:
        try:
            predicted_intent: str = model.predict([query])[0]

            # Validasi bahwa prediksi adalah intent yang dikenali
            if predicted_intent not in VALID_INTENTS:
                logger.warning(
                    f"Model predicted unknown intent '{predicted_intent}', "
                    f"falling back to 'ensiklopedia'."
                )
                return "ensiklopedia"

            logger.info(f"Intent classified via SVM model: '{predicted_intent}'")
            return predicted_intent

        except Exception as e:
            logger.error(f"SVM prediction failed: {e}", exc_info=True)
            return _keyword_fallback(query)

    return _keyword_fallback(query)


def classify_attachment_intent(query: str, attachment: str) -> str:
    """
    Mengklasifikasikan intent khusus untuk request yang memiliki attachment.
    """
    query_lower = query.lower()

    if any(k in query_lower for k in ["tanaman apa", "molekul apa", "senyawa apa", "herbal apa", "nama senyawa", "struktur apa", "what compound", "what molecule", "dari tanaman"]):
        return "identify_compound_from_attachment"

    if any(k in query_lower for k in ["identifikasi", "identify", "senyawa", "molekul", "compound", "molecule"]):
        return "identify_compound"

    if any(k in query_lower for k in ["ringkaskan", "rangkum", "summarize", "summary", "ikhtisar"]):
        return "summarize_document"

    if any(k in query_lower for k in ["tabel", "table", "kolom", "column", "baris", "row"]):
        return "analyze_table"

    if any(k in query_lower for k in ["rumus", "persamaan", "formula", "reaksi", "equation"]):
        return "analyze_formula"

    if any(k in query_lower for k in ["ekstrak", "ambil data", "extract"]):
        return "extract_document"

    if any(k in query_lower for k in ["apa", "bagaimana", "mengapa", "tanya", "siapa", "kapan", "?", "how", "what", "why"]):
        return "question_answering_document"

    return "analyze_attachment"
