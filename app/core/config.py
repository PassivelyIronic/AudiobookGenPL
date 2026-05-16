"""
Centralna konfiguracja aplikacji.

Wszystkie parametry można nadpisać przez zmienne środowiskowe albo plik
`.env` w katalogu głównym projektu. Konwencja: prefix `AK_` ("Audio
Książnica"), żeby nie kolidować z innymi env-ami systemowymi.

Przykład `.env`:
    AK_REDIS_URL=redis://localhost:6379/0
    AK_TTS_MOCK_MODE=false
    AK_TTS_SPEAKER_WAV=/data/voices/lektor.wav
    AK_MAX_UPLOAD_SIZE_MB=200
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Pojedyncze źródło prawdy o konfiguracji - czytane z env/.env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="AK_",
        extra="ignore",
    )

    # ----- Redis / Celery ---------------------------------------------------
    redis_url: str = "redis://localhost:6379/0"
    celery_task_time_limit_sec: int = 3600 * 6  # 6h - audiobook dla grubej książki
    celery_result_ttl_sec: int = 3600 * 24      # 24h - tyle przechowujemy wyniki

    # ----- Składnica plików -------------------------------------------------
    upload_dir: Path = Path("./storage/uploads")
    output_dir: Path = Path("./storage/outputs")
    work_dir: Path = Path("./storage/work")
    cleanup_upload_after_success: bool = True
    cleanup_work_dir_after_success: bool = True

    # ----- TTS --------------------------------------------------------------
    tts_mock_mode: bool = True
    tts_language: str = "pl"
    tts_speaker_wav: Path | None = None
    tts_device: str = "cuda"

    # ----- Chunker ----------------------------------------------------------
    chunk_max_chars: int = Field(default=500, ge=20, le=2000)

    # ----- Stitcher ---------------------------------------------------------
    mp3_bitrate: str = "192k"
    mp3_sample_rate: int = 24_000

    # ----- API --------------------------------------------------------------
    max_upload_size_mb: int = Field(default=100, ge=1, le=1000)
    allowed_upload_extensions: tuple[str, ...] = (".epub",)

    # ----- Helpery ----------------------------------------------------------

    def ensure_directories(self) -> None:
        """Tworzy katalogi roboczo-składowe, jeśli nie istnieją."""
        for directory in (self.upload_dir, self.output_dir, self.work_dir):
            directory.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Cache'owana fabryka settingsów - jedna instancja na cały proces.

    Używać w FastAPI poprzez Depends(get_settings), w Celery i pipeline
    importować bezpośrednio. Cache można sprzątnąć w testach przez
    `get_settings.cache_clear()`.
    """
    return Settings()
