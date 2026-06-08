from app.core.config import Settings


def test_settings_accepts_app_name(monkeypatch):
    monkeypatch.setenv(
        "APP_NAME",
        "Enterprise GraphRAG Agentic AI",
    )

    settings = Settings()

    assert settings.APP_NAME == "Enterprise GraphRAG Agentic AI"


def test_extra_environment_does_not_crash(monkeypatch):
    monkeypatch.setenv(
        "UNUSED_LEGACY_SETTING",
        "legacy-value",
    )

    settings = Settings()

    assert settings is not None


def test_minio_endpoint_is_sanitized(monkeypatch):
    monkeypatch.setenv(
        "MINIO_ENDPOINT",
        "http://medical-minio:9000/path",
    )

    settings = Settings()

    assert settings.MINIO_ENDPOINT == "medical-minio:9000"


def test_config_module_import():
    from app.core.config import settings

    assert settings.APP_NAME


def test_ocr_worker_defaults():
    settings = Settings()

    assert settings.OCR_WORKER_URL == "http://medical_ocr_worker:8010"
    assert settings.OCR_WORKER_TIMEOUT_SECONDS == 180
    assert settings.OCR_WORKER_ENABLED is True
