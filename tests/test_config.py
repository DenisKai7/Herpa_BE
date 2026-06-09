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
    assert settings.VLM_ROUTER_BASE_URL == "https://router.huggingface.co/v1"
    assert settings.VLM_PRIMARY_MODEL == "zai-org/GLM-4.5V:cheapest"
    assert settings.VLM_FALLBACK_MODELS == "CohereLabs/command-a-vision-07-2025:cohere"
    assert settings.VLM_DISABLED_MODELS == "Qwen/Qwen2.5-VL-7B-Instruct"
    assert settings.vlm_model_candidates == [
        "zai-org/GLM-4.5V:cheapest",
        "CohereLabs/command-a-vision-07-2025:cohere",
    ]
    assert settings.VLM_ALLOW_MODEL_SUBSTITUTION is True
    assert settings.VLM_MAX_RETRIES_PER_MODEL == 1
    assert settings.VLM_AVAILABILITY_CACHE_SECONDS == 600
    assert settings.VLM_FAILURE_COOLDOWN_SECONDS == 600
    assert settings.ATTACHMENT_PROCESSING_ASYNC is True
