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
    service: str = "epub_narrate"
    version: str = "0.2.0"


# ============================================================
#  Kolejka zadań - widok administratora
# ============================================================


QueueState = Literal["ACTIVE", "RESERVED"]


class QueueItem(BaseModel):
    """Pojedyncze zadanie w kolejce Celery."""

    task_id: str = Field(..., description="Identyfikator zadania")
    state: QueueState = Field(
        ...,
        description=(
            "ACTIVE = aktualnie przetwarzane przez workera, "
            "RESERVED = zakolejkowane, czeka na slot"
        ),
    )
    worker: str | None = Field(
        None, description="Nazwa workera, który przejął zadanie"
    )
    name: str | None = Field(
        None, description="Nazwa funkcji taska", examples=["epub_narrate.process_epub"]
    )
    epub_filename: str | None = Field(
        None,
        description="Nazwa wgranego pliku EPUB (jeśli da się wydedukować z args)",
    )
    received_at: float | None = Field(
        None, description="Unix timestamp odebrania przez workera"
    )


class QueueResponse(BaseModel):
    """Odpowiedź endpointu /queue."""

    items: list[QueueItem] = Field(default_factory=list)
    workers_online: int = Field(
        0, ge=0, description="Liczba workerów Celery odpowiadających na ping"
    )
    broker_reachable: bool = Field(
        True, description="Czy backend Celery (Redis) jest osiągalny"
    )
