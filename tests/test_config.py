from app.core.config import Settings


def test_settings_accepts_app_name(monkeypatch):
    monkeypatch.setenv("APP_NAME", "Enterprise GraphRAG Agentic AI")

    settings = Settings()

    assert settings.APP_NAME == "Enterprise GraphRAG Agentic AI"


def test_extra_environment_does_not_crash(monkeypatch):
    monkeypatch.setenv("UNUSED_LEGACY_SETTING", "legacy-value")

    settings = Settings()

    assert settings is not None


def test_minio_endpoint_is_sanitized(monkeypatch):
    monkeypatch.setenv("MINIO_ENDPOINT", "http://medical-minio:9000/path")

    settings = Settings()

    assert settings.MINIO_ENDPOINT == "medical-minio:9000"


def test_config_module_import():
    from app.core.config import settings

    assert settings.APP_NAME


def test_vlm_defaults():
    settings = Settings()

    assert settings.VLM_BACKEND == "hf_router"
    assert settings.VLM_MODEL_ID == "Qwen/Qwen2.5-VL-7B-Instruct"
    assert settings.VLM_ROUTER_BASE_URL == "https://router.huggingface.co/v1"
    assert settings.VLM_ALLOW_MODEL_SUBSTITUTION is False
    assert settings.ATTACHMENT_PROCESSING_ASYNC is True
