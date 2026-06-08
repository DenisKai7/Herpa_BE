"""Optional GOT-OCR2 preload command. Not used during Docker build."""

from __future__ import annotations

import asyncio

from app.ocr_worker.service import got_ocr_service


async def main() -> None:
    await got_ocr_service.ensure_loaded()
    print("GOT-OCR2 model preloaded")


if __name__ == "__main__":
    asyncio.run(main())
