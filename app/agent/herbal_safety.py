"""Deterministic safety checks for herbal recommendations."""

from app.models.herbal_recommendation import HerbalRecommendationRequest, ExtractedComplaint

RED_FLAG_TERMS = {
    "sesak napas": "Sesak napas dapat menandakan kondisi serius.",
    "nyeri dada": "Nyeri dada perlu evaluasi medis segera.",
    "bibir membengkak": "Bengkak pada bibir dapat menandakan reaksi alergi berat.",
    "wajah membengkak": "Bengkak pada wajah dapat menandakan reaksi alergi berat.",
    "sulit menelan": "Sulit menelan dapat menandakan reaksi alergi atau sumbatan serius.",
    "penurunan kesadaran": "Penurunan kesadaran adalah tanda bahaya.",
    "tidak sadar": "Penurunan kesadaran adalah tanda bahaya.",
    "kejang": "Kejang membutuhkan evaluasi tenaga kesehatan.",
    "perdarahan": "Perdarahan perlu pemeriksaan medis.",
    "muntah darah": "Muntah darah adalah tanda bahaya.",
    "bab hitam": "BAB hitam dapat menandakan perdarahan saluran cerna.",
    "demam tinggi": "Demam tinggi menetap perlu evaluasi medis.",
    "dehidrasi": "Dehidrasi perlu pertolongan medis.",
    "dehidrasi berat": "Dehidrasi berat perlu pertolongan medis.",
    "keluhan memburuk": "Keluhan yang memburuk perlu evaluasi medis.",
    "memburuk": "Keluhan yang memburuk perlu evaluasi medis.",
    "kelemahan mendadak": "Kelemahan mendadak perlu pemeriksaan segera.",
}
MEDICAL_ATTENTION_SIGNS = [
    "sesak napas",
    "nyeri dada",
    "bibir atau wajah membengkak",
    "sulit menelan",
    "penurunan kesadaran",
    "kejang",
    "perdarahan",
    "muntah darah",
    "BAB hitam",
    "demam tinggi menetap",
    "dehidrasi",
    "keluhan memburuk",
    "keluhan pada bayi",
    "keluhan serius pada kehamilan",
]
AMBIGUOUS_TERMS = {"panas dalam", "sakit", "tidak enak badan", "radang", "masuk angin"}
CLARIFICATION_QUESTIONS = [
    "Keluhan utama terasa di bagian mana?",
    "Sudah berapa lama?",
    "Apakah disertai demam?",
    "Apakah ada sesak napas?",
    "Apakah sulit menelan?",
    "Apakah sedang hamil atau mengonsumsi obat rutin?",
]


def medical_attention_signs() -> list[str]:
    return MEDICAL_ATTENTION_SIGNS.copy()


def deterministic_red_flags(req: HerbalRecommendationRequest, extracted: ExtractedComplaint) -> list[str]:
    text = req.complaint.lower()
    flags = list(extracted.red_flags)
    for term in RED_FLAG_TERMS:
        if term in text and term not in flags:
            flags.append(term)
    if req.age_group == "infant":
        flags.append("keluhan pada bayi")
    if req.pregnancy_status == "pregnant" and extracted.severity in {"moderate", "severe"}:
        flags.append("keluhan serius pada kehamilan")
    return list(dict.fromkeys(flags))


def needs_clarification(req: HerbalRecommendationRequest, extracted: ExtractedComplaint) -> bool:
    text = req.complaint.lower().strip()
    if text in AMBIGUOUS_TERMS:
        return True
    if len(extracted.primary_symptoms) == 0 and any(term in text for term in AMBIGUOUS_TERMS):
        return True
    return False


def _contains_any(values: list[str], needles: list[str]) -> bool:
    haystack = " | ".join(values).lower()
    return any(needle and needle.lower() in haystack for needle in needles)


def safety_assess(raw: dict, req: HerbalRecommendationRequest) -> tuple[str, list[str]]:
    reasons: list[str] = []
    status = "eligible"
    all_safety = []
    for key in ["contraindications", "interactions", "side_effects", "risk_groups", "warnings", "toxicity"]:
        for value in raw.get(key, []) or []:
            if isinstance(value, dict):
                all_safety.extend(str(value.get(k)) for k in ["title", "description", "name", "label"] if value.get(k))
            elif value:
                all_safety.append(str(value))

    names = [raw.get("local_name", ""), raw.get("scientific_name", ""), *raw.get("aliases", [])]
    if _contains_any(names, req.allergies):
        return "excluded", ["Kandidat dikeluarkan karena cocok dengan data alergi pengguna."]

    safety_text = " | ".join(all_safety).lower()
    if req.pregnancy_status in {"pregnant", "breastfeeding"} and any(k in safety_text for k in ["hamil", "kehamilan", "menyusui", "pregnan", "laktasi"]):
        status = "excluded" if req.pregnancy_status == "pregnant" else "conditional"
        reasons.append("Terdapat data kewaspadaan untuk kehamilan atau menyusui.")
    if req.age_group == "child" and any(k in safety_text for k in ["anak", "child"]):
        status = "conditional"
        reasons.append("Terdapat batasan atau kewaspadaan untuk anak.")
    if req.age_group == "elderly":
        status = "conditional"
        reasons.append("Lansia perlu skrining penyakit penyerta dan interaksi obat.")
    if req.current_medications and raw.get("interactions"):
        status = "conditional" if status != "excluded" else status
        reasons.append("Ada data interaksi; perlu disesuaikan dengan obat rutin pengguna.")
    if not all_safety:
        status = "conditional" if status == "eligible" else status
        reasons.append("Data keamanan belum lengkap pada knowledge graph.")
    return status, reasons


def medical_attention_message(red_flags: list[str]) -> str:
    joined = ", ".join(red_flags)
    return (
        f"Ditemukan tanda kewaspadaan: {joined}. Jangan menunda pertolongan. "
        "Rekomendasi herbal mandiri tidak ditampilkan; segera hubungi tenaga kesehatan atau layanan gawat darurat bila gejala berat."
    )
