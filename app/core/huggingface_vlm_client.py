import asyncio
import base64
import io
import json
import logging
import time
from typing import Any

import httpx
from PIL import Image, ImageOps

logger = logging.getLogger(__name__)


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
        "model_not_supported": "Model visual belum tersedia pada Hugging Face Inference Provider yang aktif.",
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
        self.client = httpx.AsyncClient(
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
        image_bytes: bytes,
        mime_type: str,
        question: str,
        system_prompt: str,
    ) -> dict:
        data_uri = create_data_uri(
            image_bytes,
            mime_type,
        )

        payload = {
            "model": self.settings.VLM_MODEL_ID,
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

        for attempt in range(
            self.settings.VLM_MAX_RETRIES + 1
        ):
            try:
                response = await self.client.post(
                    self.endpoint_url(),
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
                            self.settings.VLM_MODEL_ID,
                        ),
                        "usage": result.get("usage"),
                    }

                code = classify_vlm_error(
                    response.status_code,
                    response.text,
                )

                # Log only status code, error classification, request ID, model ID
                request_id = response.headers.get("x-request-id", "unknown")
                logger.error(
                    "VLM request failed: status_code=%s error_classification=%s request_id=%s model_id=%s",
                    response.status_code,
                    code,
                    request_id,
                    self.settings.VLM_MODEL_ID,
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

            if attempt < self.settings.VLM_MAX_RETRIES:
                await asyncio.sleep(2)

        raise last_error or HuggingFaceVlmError(
            "provider_error",
            "Layanan visual gagal.",
            retryable=True,
        )

    async def repair_json(self, *, raw_text: str, system_prompt: str) -> str:
        payload = {
            "model": self.settings.VLM_MODEL_ID,
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
                self.endpoint_url(),
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

    async def healthcheck(self) -> dict[str, Any]:
        base = {
            "backend": self.settings.VLM_BACKEND,
            "model_id": self.settings.VLM_MODEL_ID,
            "local_inference": False,
        }

        now = time.monotonic()
        cache_ttl = getattr(self.settings, "VLM_HEALTHCHECK_CACHE_SECONDS", 600)
        if self._health_cache and (now - self._health_cache[0] < cache_ttl):
            return self._health_cache[1]

        try:
            payload = {
                "model": self.settings.VLM_MODEL_ID,
                "messages": [
                    {
                        "role": "user",
                        "content": "ping"
                    }
                ],
                "max_tokens": 5,
                "temperature": 0.1,
            }

            response = await self.client.post(
                self.endpoint_url(),
                headers={
                    "Authorization": f"Bearer {self.settings.HF_API_TOKEN}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )

            if response.is_success:
                result = {**base, "available": True}
            else:
                code = classify_vlm_error(response.status_code, response.text)
                result = {**base, "available": False, "reason": code}
        except Exception as exc:
            logger.warning("VLM healthcheck connection failed: %s", exc)
            result = {**base, "available": False, "reason": "provider_error"}

        self._health_cache = (now, result)
        return result
