"""
Punkt wejścia FastAPI - aplikacja Audio-Książnica.

Uruchomienie lokalnie:
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

Uruchomienie produkcyjne:
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 2

Worker Celery (osobny proces):
    celery -A app.worker.celery_app worker --loglevel=info --concurrency=1
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

from app.api.routers import router as api_router
from app.core.config import get_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Twórz katalogi przed pierwszym requestem, sprzątaj przy shutdown."""
    settings = get_settings()
    settings.ensure_directories()
    logger.info(
        "FastAPI start. uploads=%s outputs=%s work=%s",
        settings.upload_dir,
        settings.output_dir,
        settings.work_dir,
    )
    yield
    logger.info("FastAPI shutdown.")


app = FastAPI(
    title="Audio-Książnica",
    description=(
        "Asynchroniczna konwersja plików EPUB do audiobooków MP3 "
        "z lokalnym modelem TTS (XTTS-v2)."
    ),
    version="0.2.0",
    lifespan=lifespan,
)

app.include_router(api_router)


@app.get("/", tags=["meta"], summary="Root - prosty banner")
async def root() -> dict[str, str]:
    return {
        "service": "audio-ksiaznica",
        "docs": "/docs",
        "health": "/health",
    }
