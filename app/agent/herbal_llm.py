"""Hugging Face Router adapter for herbal recommendation JSON tasks (dual-verification)."""

import json
import logging
import re
from typing import Any

from huggingface_hub import InferenceClient
from pydantic import ValidationError

from app.core.config import settings
from app.models.herbal_recommendation import ExtractedComplaint, HerbalRecommendationError

logger = logging.getLogger(__name__)

_client = InferenceClient(provider="auto", api_key=settings.HF_API_TOKEN)

SYMPTOM_EXTRACTION_PROMPT = """Anda bertugas mengekstrak gejala, bukan mendiagnosis.

Gunakan hanya informasi yang tertulis dalam keluhan pengguna.
Jangan menambahkan gejala yang tidak diberikan.
Normalisasikan istilah awam ke istilah yang dapat dicari dalam knowledge graph, tetapi pertahankan teks asli.
Tandai red flag seperti:
- sesak napas;
- nyeri dada;
- penurunan kesadaran;
- kejang;
- perdarahan;
- muntah darah;
- BAB hitam;
- demam tinggi menetap;
- dehidrasi berat;
- kelemahan mendadak;
- reaksi alergi berat;
- keluhan pada bayi;
- keluhan berat pada kehamilan.

Keluarkan JSON valid tanpa markdown dengan field:
original_text, normalized_summary, primary_symptoms, secondary_symptoms, body_systems, duration_text, severity, red_flags, possible_intents, requires_medical_evaluation, clarification_questions.
severity hanya: unknown, mild, moderate, severe."""

GROUNDED_EXPLANATION_PROMPT = """Anda adalah formatter rekomendasi herbal.

Semua fakta berasal dari GRAPH_VERIFIED_CONTEXT.
MODEL_THINKING hanya menyusun bahasa. Neo4j menentukan fakta. Safety engine menentukan kelayakan.

Jangan menambah, mengurangi, atau mengubah:
- bahan;
- angka;
- takaran;
- langkah;
- frekuensi;
- durasi;
- aturan pakai;
- kontraindikasi;
- interaksi;
- efek samping;
- peringatan;
- sumber.

Gunakan Bahasa Indonesia yang mudah dipahami.
Gunakan istilah "membantu meredakan" atau "terapi penunjang", bukan "menyembuhkan".
Jangan memberikan diagnosis.

Output JSON valid tanpa markdown. Gunakan candidate_id/canonical_key yang diberikan; jangan membuat ID baru dan jangan menggandakan ID:
{"candidate_explanations":[{"candidate_id":"<candidate_id>","summary":"penjelasan singkat berdasarkan GRAPH_VERIFIED_CONTEXT"}]}"""

# ---------------------------------------------------------------------------
# Model generator prompt for non-critical field completion
# ---------------------------------------------------------------------------

MODEL_GENERATOR_PROMPT = """Anda adalah modul pelengkap informasi non-kritis untuk rekomendasi tanaman herbal.

Identitas tanaman dan gejala berasal dari Neo4j Knowledge Graph.

Anda hanya boleh melengkapi informasi non-kritis yang ditandai MODEL_ALLOWED_FIELDS.

Anda dilarang membuat:

- dosis numerik;
- jumlah konsumsi;
- frekuensi konsumsi;
- durasi terapi;
- kontraindikasi spesifik;
- interaksi obat;
- efek samping spesifik;
- saran untuk kehamilan;
- saran untuk bayi;
- saran mengganti obat dokter;
- klaim menyembuhkan.

Gunakan istilah:
- membantu meredakan;
- terapi penunjang;
- penggunaan tradisional;
- bukti klinis masih terbatas.

Jika informasi tidak dapat diberikan secara aman, gunakan null.

Keluarkan JSON valid tanpa markdown."""

# ---------------------------------------------------------------------------
# Model critic prompt for safety validation
# ---------------------------------------------------------------------------

MODEL_CRITIC_PROMPT = """Anda adalah validator keselamatan output rekomendasi herbal.

Periksa MODEL_GENERATED_DATA terhadap GRAPH_CONTEXT.

Tolak output apabila:

- menyebut tanaman berbeda;
- menambah senyawa yang tidak ada;
- membuat dosis;
- membuat frekuensi;
- membuat durasi;
- membuat interaksi obat;
- membuat kontraindikasi;
- membuat klaim menyembuhkan;
- menyatakan aman untuk semua orang;
- bertentangan dengan data Neo4j;
- mengubah angka atau fakta graph.

Keluarkan JSON:

{
  "passed": true,
  "violations": [],
  "safe_fields": [],
  "rejected_fields": [],
  "confidence": 0.0
}"""


def _json_from_text(text: str) -> Any:
    cleaned = text.strip()
    if "```" in cleaned:
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
        if match:
            cleaned = match.group(1).strip()
    return json.loads(cleaned)


def _chat_json(system_prompt: str, user_payload: str, max_tokens: int | None = None) -> Any:
    try:
        res = _client.chat.completions.create(
            model=settings.HERBAL_RECOMMENDATION_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_payload},
            ],
            temperature=settings.HERBAL_RECOMMENDATION_TEMPERATURE,
            max_tokens=max_tokens or settings.HERBAL_RECOMMENDATION_MAX_TOKENS,
        )
        content = res.choices[0].message.content
        if not content:
            raise ValueError("empty model response")
        return _json_from_text(content)
    except Exception as exc:
        raise HerbalRecommendationError(
            "HERBAL_MODEL_UNAVAILABLE",
            "Layanan model ekstraksi gejala herbal sedang tidak tersedia.",
            status_code=503,
            retryable=True,
        ) from exc


def extract_complaint(complaint: str) -> ExtractedComplaint:
    payload = f"Keluhan pengguna: {complaint}"
    try:
        data = _chat_json(SYMPTOM_EXTRACTION_PROMPT, payload)
        return ExtractedComplaint.model_validate(data)
    except (ValidationError, HerbalRecommendationError, json.JSONDecodeError) as first_error:
        if isinstance(first_error, HerbalRecommendationError) and first_error.code == "HERBAL_MODEL_UNAVAILABLE":
            raise
        try:
            repair_prompt = (
                SYMPTOM_EXTRACTION_PROMPT
                + "\nPerbaiki output sebelumnya menjadi JSON valid sesuai schema. Jangan tambah informasi baru."
            )
            data = _chat_json(repair_prompt, payload)
            return ExtractedComplaint.model_validate(data)
        except Exception as exc:
            logger.info("herbal_recommendation_failed stage=symptom_extraction error_code=HERBAL_SYMPTOM_EXTRACTION_FAILED retryable=false")
            raise HerbalRecommendationError(
                "HERBAL_SYMPTOM_EXTRACTION_FAILED",
                "Keluhan belum dapat diekstrak menjadi gejala terstruktur.",
                status_code=502,
                retryable=False,
            ) from exc


def _parse_explanation_response(data: Any, allowed_ids: set[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    if isinstance(data, dict) and isinstance(data.get("candidate_explanations"), list):
        for item in data["candidate_explanations"]:
            if not isinstance(item, dict):
                continue
            candidate_id = str(item.get("candidate_id") or "")
            if candidate_id not in allowed_ids or candidate_id in result:
                continue
            text = str(item.get("summary") or item.get("reason") or "").strip()
            if text:
                result[candidate_id] = text
        return result
    explanations = data.get("explanations", {}) if isinstance(data, dict) else {}
    if isinstance(explanations, dict):
        for key, value in explanations.items():
            candidate_id = str(key)
            if candidate_id in allowed_ids and candidate_id not in result:
                result[candidate_id] = str(value)
    return result


def _candidate_batches(candidates: list[dict[str, Any]], size: int = 10) -> list[list[dict[str, Any]]]:
    return [candidates[index:index + size] for index in range(0, len(candidates), size)]


def build_grounded_explanations(context: dict[str, Any]) -> dict[str, str]:
    candidates = context.get("candidates") or []
    if not candidates:
        return {}
    result: dict[str, str] = {}
    for batch in _candidate_batches(candidates, 10):
        batch_context = {**context, "candidates": batch}
        allowed_ids = {str(item.get("candidate_id") or item.get("canonical_key") or "") for item in batch}
        payload = "GRAPH_VERIFIED_CONTEXT:\n" + json.dumps(batch_context, ensure_ascii=False)
        try:
            data = _chat_json(GROUNDED_EXPLANATION_PROMPT, payload)
            result.update(_parse_explanation_response(data, allowed_ids))
        except HerbalRecommendationError:
            raise
        except Exception:
            continue
    return result


# ---------------------------------------------------------------------------
# Dual-verification: model generator for non-critical fields
# ---------------------------------------------------------------------------

def model_generate_noncritical_fields(
    herb_context: dict[str, Any],
    complaint: str,
    matched_symptoms: list[str],
    graph_context: dict[str, Any],
    missing_fields: list[str],
    allowed_fields: list[str],
) -> dict[str, Any] | None:
    """Call MODEL_THINKING to fill non-critical missing fields.

    Returns parsed JSON dict or None on failure.
    """
    payload = json.dumps({
        "herb": {
            "local_name": herb_context.get("local_name", ""),
            "scientific_name": herb_context.get("scientific_name", ""),
        },
        "complaint": complaint,
        "matched_symptoms": matched_symptoms,
        "graph_context": graph_context,
        "missing_fields": missing_fields,
        "model_allowed_fields": allowed_fields,
    }, ensure_ascii=False)

    try:
        result = _chat_json(MODEL_GENERATOR_PROMPT, payload, max_tokens=1200)
        if isinstance(result, dict):
            return result
        return None
    except HerbalRecommendationError:
        logger.warning("herbal_model_generator_failed herb=%s", herb_context.get("local_name", "unknown"))
        return None
    except Exception:
        logger.warning("herbal_model_generator_unexpected_error herb=%s", herb_context.get("local_name", "unknown"))
        return None


# ---------------------------------------------------------------------------
# Dual-verification: model critic for safety validation
# ---------------------------------------------------------------------------

def model_critic_validate(
    model_generated_data: dict[str, Any],
    graph_context: dict[str, Any],
    herb_name: str,
) -> dict[str, Any]:
    """Call MODEL_THINKING as a critic to validate generated data.

    Returns: {"passed": bool, "violations": list, "safe_fields": list,
              "rejected_fields": list, "confidence": float}
    """
    default_reject = {
        "passed": False,
        "violations": ["critic_unavailable"],
        "safe_fields": [],
        "rejected_fields": list(model_generated_data.keys()),
        "confidence": 0.0,
    }

    payload = json.dumps({
        "MODEL_GENERATED_DATA": model_generated_data,
        "GRAPH_CONTEXT": graph_context,
        "herb_name": herb_name,
    }, ensure_ascii=False)

    try:
        result = _chat_json(MODEL_CRITIC_PROMPT, payload, max_tokens=800)
        if not isinstance(result, dict):
            return default_reject
        # Validate critic output structure
        passed = result.get("passed", False)
        confidence = float(result.get("confidence", 0.0))
        violations = result.get("violations", [])
        safe_fields = result.get("safe_fields", [])
        rejected_fields = result.get("rejected_fields", [])

        if not isinstance(violations, list):
            violations = []
        if not isinstance(safe_fields, list):
            safe_fields = []
        if not isinstance(rejected_fields, list):
            rejected_fields = []

        return {
            "passed": bool(passed),
            "violations": violations,
            "safe_fields": safe_fields,
            "rejected_fields": rejected_fields,
            "confidence": max(0.0, min(1.0, confidence)),
        }
    except HerbalRecommendationError:
        logger.warning("herbal_model_critic_failed herb=%s", herb_name)
        return default_reject
    except Exception:
        logger.warning("herbal_model_critic_unexpected_error herb=%s", herb_name)
        return default_reject
