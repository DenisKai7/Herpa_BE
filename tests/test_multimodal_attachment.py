import io
import os
import sys

import pytest
from PIL import Image

sys.path.append(os.getcwd())

if "SUPABASE_URL" not in os.environ:
    os.environ["SUPABASE_URL"] = "https://example.supabase.co"
if "SUPABASE_SERVICE_KEY" not in os.environ:
    os.environ["SUPABASE_SERVICE_KEY"] = "dummykey"
if "HF_API_TOKEN" not in os.environ:
    os.environ["HF_API_TOKEN"] = "dummyhftoken"

from app.agent import multimodal
from app.agent.multimodal import (
    OcrExtractionResult,
    build_attachment_context_package,
    format_attachment_context_package,
    preprocess_image,
    process_attachment,
)


def _png_bytes() -> bytes:
    image = Image.new("RGB", (240, 120), "white")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


@pytest.mark.asyncio
async def test_image_attachment_uses_got_ocr2_and_builds_context(monkeypatch):
    async def fake_extract(image, mode="auto"):
        return OcrExtractionResult(
            success=True,
            raw_text="Curcumin C21H20O6 OH 1 2",
            normalized_text="Curcumin C21H20O6 OH 1 2",
            detected_type="chemical_structure_diagram",
            visible_labels=["OH"],
            chemical_terms=["Curcumin"],
            molecular_formulas=["C21H20O6"],
            numeric_labels=["1", "2"],
            confidence=0.62,
            model_id="stepfun-ai/GOT-OCR-2.0-hf",
            processing_ms=10,
        )

    monkeypatch.setattr(multimodal.ocr_service, "extract", fake_extract)
    monkeypatch.setattr(multimodal.settings, "NEO4J_ATTACHMENT_VERIFICATION", False)

    result = await process_attachment(
        filename="structure.png",
        mime_type="image/png",
        content=_png_bytes(),
        user_query="gambar itu molekul dari tanaman apa?",
    )

    assert result.extraction_method == "GOT-OCR2"
    assert result.structured_data["detected_type"] == "chemical_structure_diagram"
    assert "Curcumin" in result.extracted_text
    package = build_attachment_context_package(result, user_question="identifikasi")
    formatted = format_attachment_context_package(package)
    assert "[ATTACHMENT EVIDENCE]" in formatted
    assert "jangan" not in formatted.lower() or "SMILES" in formatted
    assert package.verification_status == "not_applicable"


def test_corrupt_image_is_rejected():
    with pytest.raises(ValueError):
        preprocess_image(b"not an image")


def test_chemical_structure_classifier_is_conservative():
    data = multimodal.classify_extracted_text("OH CH3 1 2 3 C21H20O6")
    assert data["detected_type"] == "chemical_structure_diagram"
    assert "OH" in data["chemical_symbols"]
