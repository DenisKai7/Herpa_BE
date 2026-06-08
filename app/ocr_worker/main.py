"""Internal FastAPI app for OCR extraction."""

from __future__ import annotations

import os

from fastapi import FastAPI, File, HTTPException, UploadFile

from app.ocr_worker.service import check_runtime, extract_file

app = FastAPI(title="Medical OCR Worker", version="1.0.0")


@app.get("/health")
def health() -> dict:
    return check_runtime()


@app.post("/internal/ocr/extract")
async def extract(file: UploadFile = File(...)) -> dict:
    filename = file.filename or "unknown"
    mime_type = (file.content_type or "").split(";", 1)[0].strip().lower()
    content = await file.read()
    try:
        return await extract_file(filename=filename, mime_type=mime_type, content=content)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
