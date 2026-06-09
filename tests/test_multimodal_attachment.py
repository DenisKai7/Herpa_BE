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
from app.core.huggingface_vlm_client import (
    classify_hf_vlm_error,
    availability_cache,
    model_health_cache,
    REMOTE_VLM_REGISTRY,
    resolve_vlm_candidates,
)


def _png_bytes() -> bytes:
    image = Image.new("RGB", (240, 120), "white")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


class FakeVlmClient:
    def __init__(self, failures=None):
        self.payload_images = []
        self.calls = []
        self.failures = failures or {}

    async def analyze_image(self, *, model_id=None, image_bytes, mime_type, question, system_prompt):
        from app.core.huggingface_vlm_client import HuggingFaceVlmError, create_data_uri
        self.calls.append(model_id)
        if model_id in self.failures:
            code = self.failures[model_id]
            raise HuggingFaceVlmError(code, code, retryable=False)
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
            "model": model_id or "zai-org/GLM-4.5V:cheapest",
            "usage": {"total_tokens": 10},
        }

    async def repair_json(self, *, model_id=None, raw_text, system_prompt):
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
    assert fake_client.calls == ["zai-org/GLM-4.5V:cheapest"]
    assert result.structured_data["requested_model"] == "zai-org/GLM-4.5V:cheapest"
    assert result.structured_data["used_model"] == "zai-org/GLM-4.5V:cheapest"
    assert result.structured_data["fallback_used"] is False
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


# ═══════════════════════════════════════════
# ENDPOINT & INTEGRATION TESTS
# ═══════════════════════════════════════════
from fastapi.testclient import TestClient
from app.main import app
from app.core.dependencies import verify_user
from app.api.auth import get_current_user
from app.api import upload
import uuid

def mock_verify_user() -> str:
    return "test-user-id"

def mock_get_current_user() -> dict:
    return {"id": "test-user-id", "username": "testuser", "role": "user"}

@pytest.fixture(autouse=True)
def clear_vlm_caches():
    availability_cache.clear()
    model_health_cache.clear()
    yield
    availability_cache.clear()
    model_health_cache.clear()
    set_vlm_client(None)


@pytest.fixture
def client():
    app.dependency_overrides[verify_user] = mock_verify_user
    app.dependency_overrides[get_current_user] = mock_get_current_user
    yield TestClient(app)
    app.dependency_overrides.clear()

def test_retry_route_registered():
    routes = {
        (route.path, tuple(sorted(route.methods or [])))
        for route in app.routes
    }
    assert any(
        path == "/api/files/{attachment_id}/retry" and "POST" in methods
        for path, methods in routes
    )

def test_retry_unknown_attachment(client, monkeypatch):
    monkeypatch.setattr(upload, "get_attachment_context_for_user", lambda u, a: None)

    response = client.post(f"/api/files/{uuid.uuid4()}/retry")
    assert response.status_code == 404
    body = response.json()
    assert body["detail"]["code"] == "ATTACHMENT_NOT_FOUND"

def test_retry_owned_attachment(client, monkeypatch):
    attachment_id = str(uuid.uuid4())
    stored_payload = {
        "user_id": "test-user-id",
        "attachment_id": attachment_id,
        "bucket": "test-bucket",
        "object_name": "test-file.png",
        "filename": "test-file.png",
        "mime_type": "image/png",
        "processing_status": "failed",
        "retry_count": 0,
    }

    monkeypatch.setattr(upload, "get_attachment_context_for_user", lambda u, a: stored_payload)
    monkeypatch.setattr(upload, "_update_attachment_payload", lambda u, a, updates: {**stored_payload, **updates})
    monkeypatch.setattr(upload, "_read_minio_object", lambda b, o: b"dummy-content")

    task_called = False
    def mock_add_task(func, *args, **kwargs):
        nonlocal task_called
        task_called = True

    monkeypatch.setattr(upload.BackgroundTasks, "add_task", mock_add_task)

    response = client.post(f"/api/files/{attachment_id}/retry")
    assert response.status_code == 202
    body = response.json()
    assert body["processing_status"] == "queued"
    assert body["attachment_id"] == attachment_id

def test_upload_is_queued_instantly(client, monkeypatch):
    class FakeMinio:
        def bucket_exists(self, bucket): return True
        def put_object(self, *args, **kwargs): return None
    monkeypatch.setattr(upload, "minio_client", FakeMinio())

    monkeypatch.setattr(upload, "save_attachment_context", lambda u, a, p: None)
    monkeypatch.setattr(upload, "_public_preview_url", lambda b, o: "http://fake-preview")

    task_enqueued = False
    def mock_add_task(func, *args, **kwargs):
        nonlocal task_enqueued
        task_enqueued = True
    monkeypatch.setattr(upload.BackgroundTasks, "add_task", mock_add_task)

    file_content = _png_bytes()
    response = client.post(
        "/api/files/upload",
        files={"file": ("test.png", file_content, "image/png")}
    )
    assert response.status_code == 202
    body = response.json()
    assert body["success"] is True
    assert body["attachment"]["processing_status"] == "queued"
    assert task_enqueued is True

@pytest.mark.asyncio
async def test_unsupported_model_failure(monkeypatch):
    from app.core.huggingface_vlm_client import HuggingFaceVlmError
    from app.api.upload import _process_attachment_job

    async def mock_process_attachment(*args, **kwargs):
        raise HuggingFaceVlmError("model_not_supported", "Model not supported", retryable=False)

    monkeypatch.setattr(upload, "process_attachment", mock_process_attachment)

    saved_payload = {}
    def mock_update_payload(user_id, attachment_id, updates):
        nonlocal saved_payload
        saved_payload.update(updates)
        return saved_payload

    monkeypatch.setattr(upload, "_update_attachment_payload", mock_update_payload)

    await _process_attachment_job(
        user_id="test-user-id",
        attachment_id="test-attachment-id",
        filename="test.png",
        mime_type="image/png",
        content=b"dummy",
    )

    assert saved_payload["processing_status"] == "failed"
    assert saved_payload["verification_status"] == "not_started"
    assert saved_payload["error"]["code"] == "VLM_MODEL_UNAVAILABLE"
    assert saved_payload["error"]["retryable"] is False

@pytest.mark.asyncio
async def test_neo4j_unavailable_completed(monkeypatch):
    from app.api.upload import _process_attachment_job
    from app.agent.multimodal import AttachmentAnalysisResult

    async def mock_process_attachment(*args, **kwargs):
        return AttachmentAnalysisResult(
            filename="test.png",
            mime_type="image/png",
            file_sha256="fake-sha",
            extraction_method="hf-vlm",
            extracted_text="Curcumin",
            processing_ms=10,
        )
    monkeypatch.setattr(upload, "process_attachment", mock_process_attachment)

    async def mock_verify_with_neo4j(*args, **kwargs):
        raise Exception("Neo4j connection timeout")
    monkeypatch.setattr(upload, "verify_attachment_with_neo4j", mock_verify_with_neo4j)

    saved_payload = {}
    def mock_update_payload(user_id, attachment_id, updates):
        nonlocal saved_payload
        saved_payload.update(updates)
        return saved_payload
    monkeypatch.setattr(upload, "_update_attachment_payload", mock_update_payload)

    await _process_attachment_job(
        user_id="test-user-id",
        attachment_id="test-attachment-id",
        filename="test.png",
        mime_type="image/png",
        content=b"dummy",
    )

    assert saved_payload["processing_status"] == "completed"
    assert saved_payload["verification_status"] == "unavailable"


@pytest.mark.asyncio
async def test_primary_unsupported_uses_fallback_and_metadata(monkeypatch):
    primary = "zai-org/GLM-4.5V:cheapest"
    fallback = "CohereLabs/command-a-vision-07-2025:cohere"
    fake_client = FakeVlmClient(failures={primary: "model_not_supported"})
    set_vlm_client(fake_client)
    monkeypatch.setattr(multimodal.settings, "NEO4J_ATTACHMENT_VERIFICATION", False)

    result = await process_attachment(
        filename="structure.png",
        mime_type="image/png",
        content=_png_bytes(),
        user_query="analisis",
    )

    assert fake_client.calls == [primary, fallback]
    assert availability_cache.is_unavailable(primary) is True
    assert result.structured_data["requested_model"] == primary
    assert result.structured_data["used_model"] == fallback
    assert result.structured_data["fallback_used"] is True
    assert result.structured_data["fallback_reason"] == "model_not_supported"


@pytest.mark.asyncio
async def test_attachment_job_fallback_success_completed_with_metadata(monkeypatch):
    from app.api.upload import _process_attachment_job
    from app.agent.multimodal import AttachmentAnalysisResult

    async def mock_process_attachment(*args, **kwargs):
        return AttachmentAnalysisResult(
            filename="test.png",
            mime_type="image/png",
            file_sha256="fake-sha",
            extraction_method="hf-vlm",
            extracted_text="Curcumin",
            processing_ms=10,
            structured_data={
                "requested_model": "zai-org/GLM-4.5V:cheapest",
                "used_model": "CohereLabs/command-a-vision-07-2025:cohere",
                "fallback_used": True,
                "fallback_reason": "model_not_supported",
            },
        )

    monkeypatch.setattr(upload, "process_attachment", mock_process_attachment)
    monkeypatch.setattr(upload, "verify_attachment_with_neo4j", lambda *args, **kwargs: None)

    saved_payload = {}
    def mock_update_payload(user_id, attachment_id, updates):
        nonlocal saved_payload
        saved_payload.update(updates)
        return saved_payload
    monkeypatch.setattr(upload, "_update_attachment_payload", mock_update_payload)

    await _process_attachment_job(
        user_id="test-user-id",
        attachment_id="test-attachment-id",
        filename="test.png",
        mime_type="image/png",
        content=b"dummy",
    )

    assert saved_payload["processing_status"] == "completed"
    assert saved_payload["requested_model"] == "zai-org/GLM-4.5V:cheapest"
    assert saved_payload["used_model"] == "CohereLabs/command-a-vision-07-2025:cohere"
    assert saved_payload["fallback_used"] is True
    assert saved_payload["fallback_reason"] == "model_not_supported"


@pytest.mark.asyncio
async def test_all_vlm_models_unavailable_sets_structured_error(monkeypatch):
    from app.api.upload import _process_attachment_job
    from app.core.huggingface_vlm_client import VlmModelsUnavailableError

    async def mock_process_attachment(*args, **kwargs):
        raise VlmModelsUnavailableError(
            "Seluruh remote VLM tidak tersedia.",
            failures=[{"model_id": "zai-org/GLM-4.5V:cheapest", "code": "model_not_supported"}],
        )

    monkeypatch.setattr(upload, "process_attachment", mock_process_attachment)

    saved_payload = {}
    def mock_update_payload(user_id, attachment_id, updates):
        nonlocal saved_payload
        saved_payload.update(updates)
        return saved_payload
    monkeypatch.setattr(upload, "_update_attachment_payload", mock_update_payload)

    await _process_attachment_job(
        user_id="test-user-id",
        attachment_id="test-attachment-id",
        filename="test.png",
        mime_type="image/png",
        content=b"dummy",
    )

    assert saved_payload["processing_status"] == "failed"
    assert saved_payload["verification_status"] == "not_started"
    assert saved_payload["retryable"] is True
    assert saved_payload["error"] == {
        "code": "VLM_ALL_MODELS_UNAVAILABLE",
        "message": "Seluruh layanan analisis gambar sedang tidak tersedia.",
        "retryable": True,
    }


def test_retry_all_models_in_cooldown_returns_503(client, monkeypatch):
    attachment_id = str(uuid.uuid4())
    stored_payload = {
        "user_id": "test-user-id",
        "attachment_id": attachment_id,
        "bucket": "test-bucket",
        "object_name": "test-file.png",
        "filename": "test-file.png",
        "mime_type": "image/png",
        "processing_status": "failed",
        "retry_count": 0,
    }
    monkeypatch.setattr(upload, "get_attachment_context_for_user", lambda u, a: stored_payload)
    availability_cache.mark_unavailable("zai-org/GLM-4.5V:cheapest", "model_not_supported", 600)
    availability_cache.mark_unavailable("CohereLabs/command-a-vision-07-2025:cohere", "provider_unavailable", 600)

    response = client.post(f"/api/files/{attachment_id}/retry")

    assert response.status_code == 503
    assert response.json()["detail"] == {
        "code": "VLM_ALL_MODELS_UNAVAILABLE",
        "message": "Belum ada model visual yang tersedia untuk percobaan ulang.",
        "retryable": True,
    }


@pytest.mark.asyncio
async def test_availability_cache_skips_unsupported_model(monkeypatch):
    primary = "zai-org/GLM-4.5V:cheapest"
    fallback = "CohereLabs/command-a-vision-07-2025:cohere"
    availability_cache.mark_unavailable(primary, "model_not_supported", 600)
    fake_client = FakeVlmClient()
    set_vlm_client(fake_client)
    monkeypatch.setattr(multimodal.settings, "NEO4J_ATTACHMENT_VERIFICATION", False)

    result = await process_attachment(
        filename="structure.png",
        mime_type="image/png",
        content=_png_bytes(),
        user_query="analisis",
    )

    assert fake_client.calls == [fallback]
    assert result.structured_data["requested_model"] == fallback
    assert result.structured_data["used_model"] == fallback
    assert result.structured_data["fallback_used"] is False


def test_resolver_excludes_disabled_qwen_and_cooldown_models():
    route = resolve_vlm_candidates("Qwen/Qwen2.5-VL-7B-Instruct")
    assert "Qwen/Qwen2.5-VL-7B-Instruct" not in route.candidate_models
    assert route.candidate_models == [
        "zai-org/GLM-4.5V:cheapest",
        "CohereLabs/command-a-vision-07-2025:cohere",
    ]

    availability_cache.mark_unavailable("zai-org/GLM-4.5V:cheapest", "model_not_supported", 600)
    route = resolve_vlm_candidates()
    assert route.candidate_models == ["CohereLabs/command-a-vision-07-2025:cohere"]


def test_no_local_vlm_inference_references():
    checked_files = [
        "app/agent/multimodal.py",
        "app/core/huggingface_vlm_client.py",
        "app/api/upload.py",
    ]
    forbidden = [
        "medical_ocr_worker",
        "medical_vlm_worker",
        "GOT-OCR2",
        "Groq Vision",
        "from torch",
        "import torch",
        "from transformers",
        "import transformers",
    ]
    for path in checked_files:
        with open(path, encoding="utf-8") as handle:
            content = handle.read()
        for token in forbidden:
            assert token not in content
