"""
Pipeline konwersji EPUB → MP3.

Logika połączona w jednej warstwie, niezależna od Celery / FastAPI:

    EpubParser → TextChunker → TTSModel → AudioStitcher

Pipeline strumieniuje audio: każdy chunk tekstu jest natychmiast syntezowany
do .wav na dysku, a po wszystkim FFmpeg łączy je w MP3 *bez* ładowania do
RAM. Tekst rozdziałów lokalnie mieści się w pamięci (kilka MB dla nawet
grubej książki), ale audio nie - i to jest tam, gdzie się pilnujemy.

Callback `progress_callback` służy do raportowania postępu - w pipeline
sam w sobie nic nie wie o Celery, ale Celery task podpina pod niego
`self.update_state(...)`.
"""
from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

from app.core.config import Settings
from app.core.tts_model import TTSModel
from app.services.audio_stitcher import AudioStitcher
from app.services.epub_parser import EpubParser
from app.services.text_chunker import TextChunker

logger = logging.getLogger(__name__)


# ============================================================
#  Protokół callbacka raportującego postęp
# ============================================================


class ProgressCallback(Protocol):
    """Sygnatura callbacka raportującego postęp.

    Implementacje muszą być TANIE - są wywoływane setki razy podczas
    konwersji. Celery task podpina pod to update_state(state='PROGRESS').
    """

    def __call__(
        self, *, stage: str, current: int, total: int, message: str = ""
    ) -> None: ...


def _noop_progress(*, stage: str, current: int, total: int, message: str = "") -> None:
    """Domyślny callback - milczący."""


# ============================================================
#  Wyjątek pipeline'u
# ============================================================


class PipelineError(Exception):
    """Błąd konwersji - opakowuje przyczynę z konkretnego etapu."""


# ============================================================
#  Wynik
# ============================================================


@dataclass(frozen=True)
class ConversionResult:
    output_path: Path
    chapters_processed: int
    chunks_synthesized: int
    output_size_bytes: int


# ============================================================
#  Pipeline
# ============================================================


class ConversionPipeline:
    """
    Pełen łańcuch: EPUB → tekst → chunki → WAV-y → MP3.

    Pipeline jest **stateless poza konfiguracją** - jedną instancję można
    używać dla wielu książek z rzędu (singleton TTSModel jest dzielony
    między wywołania).
    """

    # Nazwy etapów raportowane do progress_callback - jedno miejsce, łatwo
    # to konsumować po stronie API.
    STAGE_PARSING: str = "parsing"
    STAGE_SYNTHESIZING: str = "synthesizing"
    STAGE_STITCHING: str = "stitching"
    STAGE_DONE: str = "done"

    def __init__(
        self,
        settings: Settings,
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        self._settings = settings
        self._progress: ProgressCallback = progress_callback or _noop_progress

        # Komponenty stateless - tworzone raz.
        self._chunker = TextChunker(max_chars=settings.chunk_max_chars)
        self._stitcher = AudioStitcher(
            bitrate=settings.mp3_bitrate,
            sample_rate=settings.mp3_sample_rate,
            cleanup=True,
        )
        # TTSModel to singleton - dostajemy istniejącą instancję, jeśli
        # worker_process_init zrobił load() przy starcie workera.
        self._tts = TTSModel(
            mock_mode=settings.tts_mock_mode,
            language=settings.tts_language,
            speaker_wav=settings.tts_speaker_wav,
            device=settings.tts_device,
        )

    # ----- API publiczne -----------------------------------------------------

    def convert(
        self,
        epub_path: Path,
        work_dir: Path,
        output_mp3: Path,
    ) -> ConversionResult:
        """
        Uruchamia pełną konwersję.

        Args:
            epub_path: ścieżka do wejściowego pliku EPUB.
            work_dir: katalog roboczy - tu lądują tymczasowe .wav-y. Po
                sukcesie zostaje pusty (stitcher kasuje WAV-y).
            output_mp3: docelowa ścieżka pliku .mp3.

        Returns:
            ConversionResult: statystyki konwersji + ścieżka do MP3.

        Raises:
            PipelineError: gdy którykolwiek etap zawiedzie.
        """
        work_dir.mkdir(parents=True, exist_ok=True)
        output_mp3.parent.mkdir(parents=True, exist_ok=True)

        try:
            # 1. PARSING --------------------------------------------------
            self._progress(
                stage=self.STAGE_PARSING,
                current=0,
                total=0,
                message=f"Otwieram {epub_path.name}",
            )
            parser = EpubParser(epub_path)
            chapters = list(parser.iter_chapters())
            if not chapters:
                raise PipelineError(
                    f"EPUB nie zawiera żadnych rozdziałów: {epub_path}"
                )
            logger.info("Sparsowano %d rozdziałów z %s", len(chapters), epub_path.name)

            # 2. CHUNKING + TTS --------------------------------------------
            # Każdy rozdział tniemy na chunki i od razu syntezujemy do WAV.
            # Lista wav_paths trzyma tylko ŚCIEŻKI (string-i), więc RAM
            # rośnie tylko o ~100 B na chunk - dla 10 tys. chunków to 1 MB.
            wav_paths: list[Path] = []
            for ch_idx, chapter in enumerate(chapters, start=1):
                self._progress(
                    stage=self.STAGE_SYNTHESIZING,
                    current=ch_idx,
                    total=len(chapters),
                    message=f"Rozdział {ch_idx}/{len(chapters)}: {chapter.title}",
                )
                for chunk_idx, chunk_text in enumerate(
                    self._chunker.iter_chunks(chapter.text), start=1
                ):
                    wav_path = work_dir / (
                        f"ch{ch_idx:04d}_chunk{chunk_idx:05d}.wav"
                    )
                    self._tts.synthesize_chunk(chunk_text, wav_path)
                    wav_paths.append(wav_path)

            if not wav_paths:
                raise PipelineError(
                    "Po chunkowaniu i syntezie nie powstał żaden plik .wav."
                )
            logger.info("Wygenerowano %d plików .wav", len(wav_paths))

            # 3. STITCHING -------------------------------------------------
            self._progress(
                stage=self.STAGE_STITCHING,
                current=0,
                total=1,
                message=f"Łączę {len(wav_paths)} plików w MP3",
            )
            stitch = self._stitcher.stitch(wav_paths, output_mp3)

            self._progress(
                stage=self.STAGE_DONE,
                current=1,
                total=1,
                message=f"Gotowe: {output_mp3.name}",
            )

            return ConversionResult(
                output_path=stitch.output_path,
                chapters_processed=len(chapters),
                chunks_synthesized=len(wav_paths),
                output_size_bytes=stitch.output_size_bytes,
            )

        except PipelineError:
            raise  # już opisany
        except Exception as exc:  # noqa: BLE001
            # Opakowujemy WSZYSTKIE inne wyjątki w PipelineError - to klient
            # taska/api dostaje czytelny komunikat, a oryginalny stack trace
            # zostaje w __cause__.
            raise PipelineError(
                f"Konwersja zawiodła na etapie pipeline: {exc}"
            ) from exc
        finally:
            # Próba sprzątania pozostałych .wav-ów (np. po porażce w środku).
            # Stitcher po sukcesie sam je kasuje; tu czyścimy tylko, jeśli
            # coś jednak zostało (np. błąd przed wywołaniem stitch()).
            if self._settings.cleanup_work_dir_after_success and work_dir.exists():
                self._cleanup_work_dir(work_dir)

    # ----- Helpery -----------------------------------------------------------

    @staticmethod
    def _cleanup_work_dir(work_dir: Path) -> None:
        """Usuwa katalog roboczy WRAZ z zawartością. Cicho ignoruje błędy."""
        try:
            shutil.rmtree(work_dir, ignore_errors=True)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Nie udało się sprzątnąć work_dir %s: %s", work_dir, exc
            )
