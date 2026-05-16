"""
Worker Celery dla pipeline'u Audio-Książnica.

Uruchomienie:
    celery -A app.worker.celery_app worker --loglevel=info --concurrency=1

UWAGA: `--concurrency=1` jest świadome - mamy jedno GPU (RTX 4060) i jeden
model TTS w VRAM. Wiele równoległych tasków waliło by się o pamięć.
Jeśli będziesz przetwarzać kilka książek równolegle na tym samym hoście,
zwiększ concurrency po zmianie na CPU-only model albo skonfiguruj kolejki
per-GPU.

Architektura:
    1. Sygnał `worker_process_init` jest emitowany dla KAŻDEGO procesu
       workera - tam ładujemy singleton TTSModel do VRAM raz na cały
       cykl życia procesu.
    2. Task `process_epub_task` wywołuje ConversionPipeline i raportuje
       postęp przez self.update_state(state='PROGRESS').
    3. Po sukcesie task zwraca ścieżkę do MP3 (string), po porażce
       re-raise'uje, co Celery zapisuje jako state='FAILURE' z traceback.
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
    broker=_settings.celery_broker_url,
    backend=_settings.celery_result_backend,
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
    task_track_started=True,           # state='STARTED' jest aktywne
    task_send_sent_event=False,
    # --- Limity ---
    task_time_limit=_settings.celery_task_time_limit_sec,        # twardy timeout
    task_soft_time_limit=_settings.celery_task_time_limit_sec - 60,
    result_expires=_settings.celery_result_ttl_sec,
    # --- Restart procesu po N taskach ---
    # Chroni przed wyciekami VRAM przy długich seriach tasków.
    worker_max_tasks_per_child=10,
    # --- Prefetch ---
    # Bierzemy 1 task na proces - duże zadania, niski concurrency.
    worker_prefetch_multiplier=1,
)


# ============================================================
#  Inicjalizacja workera - ładowanie modelu raz przy starcie
# ============================================================


@worker_process_init.connect
def _init_worker(**kwargs: Any) -> None:
    """
    Ładuje singleton TTSModel do pamięci procesu workera.

    Sygnał jest emitowany RAZ na proces - jeśli używasz prefork-poola
    (domyślny), każdy proces dziecko dostanie własną instancję singletona.
    To jest pożądane: różne procesy = różne sesje torch.cuda.
    """
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
        logger.exception(
            "Nie udało się załadować modelu TTS przy starcie workera"
        )
        # Re-raise - worker NIE powinien startować bez modelu.
        raise
    logger.info("Worker gotowy.")


# ============================================================
#  Task: konwersja EPUB → MP3
# ============================================================


@celery_app.task(bind=True, name="epub_narrate.process_epub") # Było audio_ksiaznica...
def process_epub_task(self, task_id: str, epub_filename: str) -> dict[str, Any]:
    """
    Konwertuje EPUB do MP3.

    Args:
        epub_path: bezwzględna ścieżka do pliku EPUB (zapisanego przez
            endpoint /upload).

    Returns:
        dict: {
            "output_path": "/abs/path/to/audiobook.mp3",
            "chapters_processed": int,
            "chunks_synthesized": int,
            "output_size_bytes": int,
        }

    Raises:
        PipelineError: gdy któryś etap konwersji zawiedzie. Celery zapisze
            to jako state='FAILURE' z pełnym tracebackiem.
    """
    settings = get_settings()
    task_id = self.request.id
    epub = Path(epub_path).resolve()

    logger.info("Task %s rozpoczęty: %s", task_id, epub)

    if not epub.is_file():
        raise PipelineError(f"Plik EPUB nie istnieje: {epub}")

    # Każdy task ma swój katalog roboczy, żeby pliki .wav z różnych
    # tasków się nie mieszały (np. gdy concurrency > 1 w przyszłości).
    work_dir = settings.work_dir / task_id
    output_mp3 = settings.output_dir / f"{epub.stem}__{task_id}.mp3"

    # Callback raportujący postęp do Celery result backend.
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
        # Re-raise - Celery zapisze FAILURE
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("Task %s padł na nieoczekiwanym błędzie", task_id)
        raise PipelineError(f"Niespodziewany błąd: {exc}") from exc
    finally:
        # Pliki uploadowane sprzątamy zawsze (nie potrzebujemy ich już po
        # zakończeniu konwersji - sukces lub porażka).
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
