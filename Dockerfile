# ═══════════════════════════════════════════
# Dockerfile - Medical AI Backend (FastAPI)
# Multi-stage build for production
# ═══════════════════════════════════════════

FROM python:3.11-slim AS base

# Metadata
LABEL maintainer="Medical AI Team"
LABEL description="Enterprise GraphRAG Agentic AI Backend"

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies for OCR (Tesseract) dan build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-ind \
    tesseract-ocr-eng \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# ── Install Python dependencies ──
COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=300 --retries=5 --upgrade pip && \
    pip install --no-cache-dir --default-timeout=300 --retries=5 -r requirements.txt

# ── Copy application code ──
COPY . .

# ── Pre-download embedding model (tidak diperlukan lagi, pakai HuggingFace Inference API) ──
# Model embedding dan LLM sekarang diakses via HuggingFace cloud API

# ── Expose port ──
EXPOSE 8000

# ── Health check ──
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# ── Run server ──
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
