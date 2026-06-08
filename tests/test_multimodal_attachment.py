import io
import os
import sys

import pytest
from PIL import Image

sys.path.append(os.getcwd())

os.environ.setdefault("SUPABASE_URL", "https://example.supabase.co")
os.environ.setdefault("SUPABASE_SERVICE_KEY", "dummykey")
os.environ.setdefault("HF_API_TOKEN", "dummyhftoken")

from app.agent import multimodal
from app.agent.multimodal import (
    build_attachment_context_package,
    format_attachment_context_package,
    preprocess_image,
    process_attachment,
    set_vlm_client,
    to_data_uri,
)
from app.core.huggingface_vlm_client import classify_hf_vlm_error


def _png_bytes() -> bytes:
    image = Image.new("RGB", (240, 120), "white")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


class FakeVlmClient:
    def __init__(self):
        self.payload_images = []

    async def analyze_image(self, *, image_bytes, mime_type, question, system_prompt):
        from app.core.huggingface_vlm_client import create_data_uri
        data_uri = create_data_uri(image_bytes, mime_type)

        class MockImage:
            def __init__(self, uri):
                self.data_uri = uri

        self.payload_images = [MockImage(data_uri)]
        return {
            "content": """{
                "detected_content_type": "chemical_structure_diagram",
                "visual_description": "Curcumin C21H20O6 OH 1 2",
                "extracted_text": "Curcumin C21H20O6 OH 1 2",
                "plant_names": ["Curcuma longa"],
                "compound_names": ["Curcumin"],
                "molecular_formulas": ["C21H20O6"],
                "visible_labels": ["OH"],
                "claims": [],
                "uncertainties": [],
                "confidence": 0.62
            }""",
            "model": "Qwen/Qwen2.5-VL-7B-Instruct",
            "usage": {"total_tokens": 10},
        }

    async def repair_json(self, *, raw_text, system_prompt):
        return raw_text


@pytest.mark.asyncio
async def test_image_attachment_uses_hf_vlm_and_builds_context(monkeypatch):
    fake_client = FakeVlmClient()
    set_vlm_client(fake_client)
    monkeypatch.setattr(multimodal.settings, "NEO4J_ATTACHMENT_VERIFICATION", False)

    result = await process_attachment(
        filename="structure.png",
        mime_type="image/png",
        content=_png_bytes(),
        user_query="gambar itu molekul dari tanaman apa?",
    )

    assert result.extraction_method == "hf-vlm"
    assert result.structured_data["detected_type"] == "chemical_structure_diagram"
    assert "Curcumin" in result.extracted_text
    assert fake_client.payload_images[0].data_uri.startswith("data:image/jpeg;base64,")
    assert "medical-minio" not in fake_client.payload_images[0].data_uri
    package = build_attachment_context_package(result, user_question="identifikasi")
    formatted = format_attachment_context_package(package)
    assert "[ATTACHMENT EVIDENCE]" in formatted
    assert "SMILES" in formatted
    assert package.verification_status == "not_applicable"
    set_vlm_client(None)


def test_corrupt_image_is_rejected():
    with pytest.raises(ValueError):
        preprocess_image(b"not an image")


def test_chemical_structure_classifier_is_conservative():
    data = multimodal.classify_extracted_text("OH CH3 1 2 3 C21H20O6")
    assert data["detected_type"] == "chemical_structure_diagram"
    assert "OH" in data["chemical_symbols"]


def test_to_data_uri_encodes_base64():
    uri = to_data_uri(b"abc", "image/png")
    assert uri == "data:image/png;base64,YWJj"


def test_hf_error_classifier():
    assert classify_hf_vlm_error(401, "bad token") == "authentication_failed"
    assert classify_hf_vlm_error(400, "model_not_supported") == "model_not_supported"
    assert classify_hf_vlm_error(400, "not supported by any provider") == "model_not_supported"
