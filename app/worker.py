"""
Worker Celery dla pipeline'u EpubNarrate.

Uruchomienie:
    celery -A app.worker.celery_app worker --loglevel=info --concurrency=1
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from celery import Celery
from celery.signals import worker_process_init
from celery.utils.log import get_task_logger

from app.core.config import get_settings
from app.core.tts_model import TTSModel
from app.services.pipeline import ConversionPipeline, PipelineError

logger = get_task_logger(__name__)

# ============================================================
#  Aplikacja Celery
# ============================================================

_settings = get_settings()

celery_app = Celery(
    "epub_narrate",
    broker=_settings.redis_url,
    backend=_settings.redis_url,
)

celery_app.conf.update(
    # --- Serializacja ---
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    # --- Strefa czasowa ---
    timezone="Europe/Warsaw",
    enable_utc=True,
    # --- Tracking stanu ---
    task_track_started=True,
    task_send_sent_event=False,
    # --- Limity ---
    task_time_limit=_settings.celery_task_time_limit_sec,
    task_soft_time_limit=_settings.celery_task_time_limit_sec - 60,
    result_expires=_settings.celery_result_ttl_sec,
    # --- Restart procesu po N taskach ---
    worker_max_tasks_per_child=10,
    # --- Prefetch ---
    worker_prefetch_multiplier=1,
)

# ============================================================
#  Inicjalizacja workera - ładowanie modelu raz przy starcie
# ============================================================

@worker_process_init.connect
def _init_worker(**kwargs: Any) -> None:
    settings = get_settings()
    logger.info(
        "Inicjalizuję worker (mock_mode=%s, device=%s)",
        settings.tts_mock_mode,
        settings.tts_device,
    )
    settings.ensure_directories()
    try:
        TTSModel(
            mock_mode=settings.tts_mock_mode,
            language=settings.tts_language,
            speaker_wav=settings.tts_speaker_wav,
            device=settings.tts_device,
        ).load()
    except Exception:
        logger.exception("Nie udało się załadować modelu TTS przy starcie workera")
        raise
    logger.info("Worker gotowy.")

# ============================================================
#  Task: konwersja EPUB → MP3
# ============================================================

@celery_app.task(bind=True, name="epub_narrate.process_epub")
def process_epub_task(self, task_id: str, epub_filename: str) -> dict[str, Any]:
    settings = get_settings()
    task_id = self.request.id
    epub = Path(epub_filename).resolve()

    logger.info("Task %s rozpoczęty: %s", task_id, epub)

    if not epub.is_file():
        raise PipelineError(f"Plik EPUB nie istnieje: {epub}")

    work_dir = settings.work_dir / task_id
    output_mp3 = settings.output_dir / f"{epub.stem}__{task_id}.mp3"

    def report_progress(*, stage: str, current: int, total: int, message: str = "") -> None:
        self.update_state(
            state="PROGRESS",
            meta={
                "stage": stage,
                "current": current,
                "total": total,
                "percent": round(100.0 * current / total, 1) if total else 0.0,
                "message": message,
            },
        )

    try:
        pipeline = ConversionPipeline(settings, progress_callback=report_progress)
        result = pipeline.convert(epub, work_dir, output_mp3)
    except PipelineError as exc:
        logger.exception("Task %s zawiódł: %s", task_id, exc)
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("Task %s padł na nieoczekiwanym błędzie", task_id)
        raise PipelineError(f"Niespodziewany błąd: {exc}") from exc
    finally:
        if settings.cleanup_upload_after_success:
            try:
                epub.unlink(missing_ok=True)
            except OSError as exc:
                logger.warning("Nie udało się usunąć uploadu %s: %s", epub, exc)

    logger.info(
        "Task %s zakończony pomyślnie: %s (%d rozdz., %d chunków, %.1f MB)",
        task_id,
        result.output_path,
        result.chapters_processed,
        result.chunks_synthesized,
        result.output_size_bytes / 1024 / 1024,
    )

    return {
        "output_path": str(result.output_path),
        "chapters_processed": result.chapters_processed,
        "chunks_synthesized": result.chunks_synthesized,
        "output_size_bytes": result.output_size_bytes,
    }