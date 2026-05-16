"""
Modele Pydantic dla API.

Trzymane osobno od endpointów - łatwiej je reużyć (np. w SDK klienckim
albo w testach) i nie zaśmiecają routers.py.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# Stany Celery, które ujawniamy publicznie. Reszta (RETRY, REVOKED) jest
# wewnętrzna i mapowana na PENDING po naszej stronie - upraszczamy klienta.
CeleryState = Literal["PENDING", "STARTED", "PROGRESS", "SUCCESS", "FAILURE"]


class UploadResponse(BaseModel):
    """Odpowiedź endpointu /upload."""

    task_id: str = Field(..., description="Identyfikator zadania w Celery")
    status_url: str = Field(
        ...,
        description="URL endpointu do odpytywania o status zadania",
        examples=["/status/abc-123-def"],
    )
    message: str = "Konwersja rozpoczęta"


class ProgressInfo(BaseModel):
    """Szczegóły postępu, gdy state == 'PROGRESS'."""

    stage: str = Field(..., description="Etap przetwarzania", examples=["synthesizing"])
    current: int = Field(..., ge=0)
    total: int = Field(..., ge=0)
    percent: float = Field(..., ge=0.0, le=100.0)
    message: str = ""


class ConversionResultPayload(BaseModel):
    """Zwartość pola `result`, gdy state == 'SUCCESS'."""

    output_path: str = Field(..., description="Bezwzględna ścieżka do MP3 na hoście workera")
    chapters_processed: int = Field(..., ge=0)
    chunks_synthesized: int = Field(..., ge=0)
    output_size_bytes: int = Field(..., ge=0)


class TaskStatusResponse(BaseModel):
    """Odpowiedź endpointu /status/{task_id}."""

    task_id: str
    state: CeleryState

    # Dokładnie jedno z poniższych jest wypełnione w zależności od state:
    result: ConversionResultPayload | None = None
    error: str | None = None
    progress: ProgressInfo | None = None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"] = "ok"
    service: str = "audio-ksiaznica"
    version: str = "0.2.0"
