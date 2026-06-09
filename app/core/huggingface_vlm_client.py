import asyncio
import base64
import io
import json
import logging
import time
from typing import Any

import httpx
from PIL import Image, ImageOps
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Model Registry
REMOTE_VLM_REGISTRY = {
    "zai-org/GLM-4.5V:cheapest": {
        "supports_images": True,
        "enabled": True,
        "priority": 1,
    },
    "CohereLabs/command-a-vision-07-2025:cohere": {
        "supports_images": True,
        "enabled": True,
        "priority": 2,
    },
    "Qwen/Qwen2.5-VL-7B-Instruct": {
        "supports_images": True,
        "enabled": False,
        "disabled_reason": "model_not_supported",
        "priority": 99,
    },
}


class VlmModelsUnavailableError(RuntimeError):
    def __init__(self, message: str, failures: list[dict] = None) -> None:
        super().__init__(message)
        self.message = message
        self.failures = failures or []


class VlmModelRoute(BaseModel):
    requested_model: str | None
    candidate_models: list[str]


class AvailabilityCache:
    def __init__(self) -> None:
        self._cache: dict[str, tuple[float, str]] = {}

    def mark_unavailable(self, model_id: str, reason: str, ttl: int) -> None:
        expire_time = time.time() + ttl
        self._cache[model_id] = (expire_time, reason)
        logger.warning(
            "AvailabilityCache: marked %s unavailable for %d seconds. Reason: %s",
            model_id,
            ttl,
            reason,
        )

    def is_unavailable(self, model_id: str) -> bool:
        if model_id not in self._cache:
            return False
        expire_time, _ = self._cache[model_id]
        if time.time() > expire_time:
            if model_id in self._cache:
                del self._cache[model_id]
            return False
        return True

    def get_reason(self, model_id: str) -> str | None:
        if model_id not in self._cache:
            return None
        expire_time, reason = self._cache[model_id]
        if time.time() > expire_time:
            if model_id in self._cache:
                del self._cache[model_id]
            return None
        return reason

    def clear(self) -> None:
        self._cache.clear()


class ModelHealthCache:
    def __init__(self) -> None:
        self._cache: dict[str, tuple[float, bool, str | None]] = {}

    def get(self, model_id: str) -> tuple[bool, str | None] | None:
        if model_id not in self._cache:
            return None
        expire_time, available, reason = self._cache[model_id]
        if time.time() > expire_time:
            if model_id in self._cache:
                del self._cache[model_id]
            return None
        return available, reason

    def set(self, model_id: str, available: bool, reason: str | None, ttl: int) -> None:
        self._cache[model_id] = (time.time() + ttl, available, reason)

    def clear(self) -> None:
        self._cache.clear()


availability_cache = AvailabilityCache()
model_health_cache = ModelHealthCache()


def resolve_vlm_candidates(
    requested_model: str | None = None,
) -> VlmModelRoute:
    from app.core.config import settings

    candidates: list[str] = []

    if (
        requested_model
        and requested_model in REMOTE_VLM_REGISTRY
        and REMOTE_VLM_REGISTRY[requested_model].get("enabled", False)
        and REMOTE_VLM_REGISTRY[requested_model].get("supports_images", False)
        and not availability_cache.is_unavailable(requested_model)
    ):
        candidates.append(requested_model)

    for model in settings.vlm_model_candidates:
        config = REMOTE_VLM_REGISTRY.get(model, {})

        if not config.get("enabled", True):
            continue

        if not config.get("supports_images", False):
            continue

        if availability_cache.is_unavailable(model):
            continue

        if model not in candidates:
            candidates.append(model)

    if not candidates:
        raise VlmModelsUnavailableError("Tidak ada remote VLM yang aktif.")

    return VlmModelRoute(
        requested_model=requested_model,
        candidate_models=candidates,
    )


class HuggingFaceVlmError(RuntimeError):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        retryable: bool,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable
        self.status_code = status_code

    def to_payload(self) -> dict[str, Any]:
        return {
            "success": False,
            "error": {
                "code": self.code,
                "message": self.message,
                "retryable": self.retryable,
            },
        }


def classify_vlm_error(
    status_code: int,
    response_body: str,
) -> str:
    text = response_body.lower()

    if (
        "model_not_supported" in text
        or "not supported by any provider" in text
    ):
        return "model_not_supported"

    if status_code == 400:
        return "invalid_request"

    if status_code == 401:
        return "authentication_failed"

    if status_code == 403:
        return "access_denied"

    if status_code == 413:
        return "payload_too_large"

    if status_code == 429:
        return "rate_limited"

    if status_code in {502, 503}:
        return "provider_unavailable"

    if status_code == 504:
        return "provider_timeout"

    return "provider_error"


def classify_hf_vlm_error(status_code: int, response_body: str) -> str:
    return classify_vlm_error(status_code, response_body)


def safe_vlm_error_message(code: str) -> str:
    return {
        "model_not_supported": "Model visual belum tersedia pada provider Hugging Face yang aktif.",
        "invalid_request": "Permintaan analisis visual tidak valid.",
        "authentication_failed": "HF_API_TOKEN tidak valid atau sudah kedaluwarsa.",
        "access_denied": "Token tidak memiliki izin Hugging Face Inference Providers untuk model ini.",
        "payload_too_large": "Payload gambar melebihi batas provider.",
        "rate_limited": "Quota atau rate limit Hugging Face sedang tercapai.",
        "provider_unavailable": "Provider Hugging Face sedang tidak tersedia.",
        "provider_timeout": "Provider Hugging Face timeout saat memproses gambar.",
    }.get(code, "Provider Hugging Face gagal memproses analisis visual.")


def preprocess_image(
    content: bytes,
    *,
    max_pixels: int,
) -> tuple[bytes, str]:
    with Image.open(io.BytesIO(content)) as image:
        image.verify()

    with Image.open(io.BytesIO(content)) as image:
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")

        if image.width * image.height > max_pixels:
            image.thumbnail(
                (3072, 3072),
                Image.Resampling.LANCZOS,
            )

        output = io.BytesIO()
        image.save(
            output,
            format="JPEG",
            quality=88,
            optimize=True,
        )

        return output.getvalue(), "image/jpeg"


def create_data_uri(
    content: bytes,
    mime_type: str,
) -> str:
    encoded = base64.b64encode(
        content
    ).decode("ascii")

    return (
        f"data:{mime_type};base64,"
        f"{encoded}"
    )


class HuggingFaceVlmClient:
    def __init__(self, settings) -> None:
        self.settings = settings
        base_url = settings.VLM_ROUTER_BASE_URL.rstrip("/")
        if settings.VLM_BACKEND == "hf_endpoint" and settings.VLM_ENDPOINT_URL:
            endpoint = settings.VLM_ENDPOINT_URL.rstrip("/")
            if endpoint.endswith("/chat/completions"):
                base_url = endpoint[:-17]
            elif endpoint.endswith("/v1"):
                base_url = endpoint
            else:
                base_url = endpoint + "/v1"

        self.client = httpx.AsyncClient(
            base_url=base_url,
            timeout=httpx.Timeout(
                connect=settings.VLM_CONNECT_TIMEOUT_SECONDS,
                read=settings.VLM_REQUEST_TIMEOUT_SECONDS,
                write=60.0,
                pool=10.0,
            ),
            limits=httpx.Limits(
                max_connections=10,
                max_keepalive_connections=5,
                keepalive_expiry=30.0,
            ),
        )
        self._health_cache = None

    async def close(self) -> None:
        await self.client.aclose()

    async def aclose(self) -> None:
        await self.client.aclose()

    def endpoint_url(self) -> str:
        if self.settings.VLM_BACKEND == "hf_router":
            return (
                self.settings
                .VLM_ROUTER_BASE_URL
                .rstrip("/")
                + "/chat/completions"
            )

        if self.settings.VLM_BACKEND == "hf_endpoint":
            endpoint = (
                self.settings
                .VLM_ENDPOINT_URL
                .rstrip("/")
            )

            if not endpoint:
                raise HuggingFaceVlmError(
                    "endpoint_unconfigured",
                    "VLM endpoint belum dikonfigurasi.",
                    retryable=False,
                )

            if endpoint.endswith(
                "/chat/completions"
            ):
                return endpoint

            if endpoint.endswith("/v1"):
                return endpoint + "/chat/completions"

            return endpoint + "/v1/chat/completions"

        raise HuggingFaceVlmError(
            "vlm_disabled",
            "Remote VLM dinonaktifkan.",
            retryable=False,
        )

    async def analyze_image(
        self,
        *,
        model_id: str | None = None,
        image_bytes: bytes,
        mime_type: str,
        question: str,
        system_prompt: str,
    ) -> dict:
        target_model = model_id or self.settings.VLM_PRIMARY_MODEL
        data_uri = create_data_uri(
            image_bytes,
            mime_type,
        )

        payload = {
            "model": target_model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": question,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": data_uri,
                            },
                        },
                    ],
                },
            ],
            "temperature": (
                self.settings.VLM_TEMPERATURE
            ),
            "max_tokens": (
                self.settings.VLM_MAX_NEW_TOKENS
            ),
        }

        last_error = None
        max_retries = getattr(self.settings, "VLM_MAX_RETRIES_PER_MODEL", 1)
        logger.info(
            "vlm_request_started model=%s endpoint=%s",
            target_model,
            self.endpoint_url(),
        )

        for attempt in range(max_retries + 1):
            try:
                response = await self.client.post(
                    "/chat/completions",
                    headers={
                        "Authorization": (
                            "Bearer "
                            + self.settings.HF_API_TOKEN
                        ),
                        "Content-Type": (
                            "application/json"
                        ),
                    },
                    json=payload,
                )

                if response.is_success:
                    result = response.json()

                    return {
                        "content": (
                            result["choices"][0]
                            ["message"]["content"]
                        ),
                        "model": result.get(
                            "model",
                            target_model,
                        ),
                        "usage": result.get("usage"),
                    }

                code = classify_vlm_error(
                    response.status_code,
                    response.text,
                )

                request_id = response.headers.get("x-request-id", "unknown")
                log_message = (
                    "vlm_model_unavailable status_code=%s reason=%s request_id=%s model_id=%s"
                    if code == "model_not_supported"
                    else "vlm_request_failed status_code=%s error_classification=%s request_id=%s model_id=%s"
                )
                logger.warning(
                    log_message,
                    response.status_code,
                    code,
                    request_id,
                    target_model,
                )

                retryable = code in {
                    "rate_limited",
                    "provider_unavailable",
                    "provider_timeout",
                    "gateway_error",
                }

                if not retryable:
                    raise HuggingFaceVlmError(
                        code,
                        safe_vlm_error_message(code),
                        retryable=False,
                        status_code=response.status_code,
                    )

                last_error = HuggingFaceVlmError(
                    code,
                    safe_vlm_error_message(code),
                    retryable=True,
                    status_code=response.status_code,
                )

            except (
                httpx.ConnectError,
                httpx.ReadTimeout,
                httpx.RemoteProtocolError,
            ) as exc:
                last_error = HuggingFaceVlmError(
                    "network_error",
                    "Koneksi ke layanan visual gagal.",
                    retryable=True,
                )

            if attempt < max_retries:
                await asyncio.sleep(2)

        raise last_error or HuggingFaceVlmError(
            "provider_error",
            "Layanan visual gagal.",
            retryable=True,
        )

    async def repair_json(self, *, model_id: str | None = None, raw_text: str, system_prompt: str) -> str:
        target_model = model_id or self.settings.VLM_PRIMARY_MODEL
        payload = {
            "model": target_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": "Perbaiki teks berikut menjadi JSON valid saja, tanpa markdown:\n" + raw_text[:6000]
                },
            ],
            "temperature": 0.0,
            "max_tokens": min(800, self.settings.VLM_MAX_NEW_TOKENS),
        }

        try:
            response = await self.client.post(
                "/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.settings.HF_API_TOKEN}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            if response.is_success:
                result = response.json()
                return result["choices"][0]["message"]["content"]
        except Exception as exc:
            logger.warning("VLM repair_json failed: %s", exc)
        return raw_text

    async def probe_model(self, model_id: str) -> tuple[bool, str | None]:
        # 1. Check fail cooldown first
        if availability_cache.is_unavailable(model_id):
            return False, availability_cache.get_reason(model_id)

        # 2. Check static config if disabled
        config = REMOTE_VLM_REGISTRY.get(model_id, {})
        if not config.get("enabled", True):
            return False, config.get("disabled_reason", "disabled")

        # 3. Check probe cache
        cached = model_health_cache.get(model_id)
        if cached is not None:
            return cached

        # 4. Perform actual probe using a tiny image
        from PIL import Image
        import io
        img = Image.new("RGB", (64, 64), color="white")
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        tiny_image_bytes = buf.getvalue()

        try:
            # We call analyze_image with a very simple prompt
            await self.analyze_image(
                model_id=model_id,
                image_bytes=tiny_image_bytes,
                mime_type="image/jpeg",
                question="ping",
                system_prompt="respond json with success true",
            )
            model_health_cache.set(
                model_id,
                True,
                None,
                self.settings.VLM_AVAILABILITY_CACHE_SECONDS,
            )
            return True, None
        except HuggingFaceVlmError as exc:
            reason = exc.code
            model_health_cache.set(
                model_id,
                False,
                reason,
                self.settings.VLM_AVAILABILITY_CACHE_SECONDS,
            )
            availability_cache.mark_unavailable(
                model_id, reason, self.settings.VLM_FAILURE_COOLDOWN_SECONDS
            )
            return False, reason
        except Exception as exc:
            reason = "provider_error"
            model_health_cache.set(
                model_id,
                False,
                reason,
                self.settings.VLM_AVAILABILITY_CACHE_SECONDS,
            )
            availability_cache.mark_unavailable(
                model_id, reason, self.settings.VLM_FAILURE_COOLDOWN_SECONDS
            )
            return False, reason

    async def healthcheck(self) -> dict[str, Any]:
        models_info = []
        for model_id in REMOTE_VLM_REGISTRY.keys():
            available, reason = await self.probe_model(model_id)
            is_cached = (
                model_health_cache.get(model_id) is not None
                or availability_cache.is_unavailable(model_id)
            )

            info = {"model_id": model_id, "available": available}
            if available:
                info["cached"] = is_cached
            else:
                info["reason"] = reason or "disabled"
            models_info.append(info)

        return {
            "backend": self.settings.VLM_BACKEND,
            "local_inference": False,
            "primary_model": self.settings.VLM_PRIMARY_MODEL,
            "models": models_info,
        }
