"""
Audio Stitcher — łączenie chunków .wav w jeden plik .mp3 przez systemowy FFmpeg.

Dlaczego FFmpeg, a nie pydub / wave / torchaudio?
    Audiobook po polsku to typowo 8-15 godzin mowy ≈ 1-2 GB nieskompresowanego
    PCM. Wczytanie tego do RAM przekroczy nasze 4.8 GB. FFmpeg w trybie
    `concat demuxer` strumieniuje dane z dysku do dysku, używając stałej,
    minimalnej ilości RAM (zazwyczaj < 50 MB).

Strategia:
    1. Tworzymy plik tekstowy `concat_list.txt` z listą:
         file '/abs/path/chunk_001.wav'
         file '/abs/path/chunk_002.wav'
         ...
    2. Wywołujemy:
         ffmpeg -f concat -safe 0 -i concat_list.txt \\
                -c:a libmp3lame -b:a 192k -ar 24000 \\
                output.mp3
    3. Po sukcesie kasujemy wszystkie .wav-y i listę.
       Po porażce ZOSTAWIAMY pliki, żeby można było zdiagnozować problem.

Sekcja `concat demuxer` w FFmpeg pozwala łączyć pliki o tym samym formacie
bez ponownego dekodowania/enkodowania, ale ponieważ konwertujemy WAV → MP3,
i tak musi przejść przez encoder. Dla naszego workloadu to akceptowalny koszt.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Sequence

logger = logging.getLogger(__name__)


# ============================================================
#  Wyjątki
# ============================================================


class AudioStitcherError(Exception):
    """Błąd łączenia audio (brak ffmpeg, błąd procesu, brak plików itp.)."""


# ============================================================
#  Wynik operacji
# ============================================================


@dataclass(frozen=True)
class StitchResult:
    """Podsumowanie zakończonej operacji łączenia."""

    output_path: Path
    input_count: int
    output_size_bytes: int
    cleanup_performed: bool


# ============================================================
#  Stitcher
# ============================================================


class AudioStitcher:
    """
    Łączy listę plików .wav w jeden plik .mp3 za pomocą systemowego FFmpeg.

    Klasa jest bezstanowa - jedną instancję można reużywać dla wielu książek.

    Przykład:
        stitcher = AudioStitcher(bitrate="192k")
        result = stitcher.stitch(
            wav_paths=[Path("ch_001.wav"), Path("ch_002.wav"), ...],
            output_mp3=Path("audiobook.mp3"),
        )
    """

    DEFAULT_BITRATE: Final[str] = "192k"
    DEFAULT_SAMPLE_RATE: Final[int] = 24_000  # matchuje XTTS-v2
    CONCAT_LIST_NAME: Final[str] = "concat_list.txt"

    def __init__(
        self,
        ffmpeg_bin: str = "ffmpeg",
        bitrate: str = DEFAULT_BITRATE,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        cleanup: bool = True,
    ) -> None:
        """
        Args:
            ffmpeg_bin: nazwa lub ścieżka do binarki ffmpeg (zwykle 'ffmpeg').
            bitrate: bitrate MP3 (np. '192k', '128k', '320k').
            sample_rate: częstotliwość próbkowania wyjścia w Hz.
            cleanup: czy usuwać pliki .wav i concat_list po sukcesie.
                Wyłącz dla diagnostyki - po porażce pliki są ZAWSZE zachowane.
        """
        self._ffmpeg_bin: str = ffmpeg_bin
        self._bitrate: str = bitrate
        self._sample_rate: int = sample_rate
        self._cleanup: bool = cleanup

    # ----- API publiczne -----------------------------------------------------

    def stitch(
        self,
        wav_paths: Sequence[str | Path],
        output_mp3: str | Path,
    ) -> StitchResult:
        """
        Łączy `wav_paths` w jeden plik `output_mp3`.

        Args:
            wav_paths: lista (lub krotka) ścieżek do plików .wav, w kolejności
                odtwarzania.
            output_mp3: docelowa ścieżka pliku .mp3.

        Returns:
            StitchResult: ścieżka wyjścia + statystyki.

        Raises:
            AudioStitcherError: gdy ffmpeg nie istnieje, lista jest pusta,
                brakuje któregokolwiek pliku wejściowego lub ffmpeg zwróci
                niezerowy kod wyjścia.
        """
        # 1. Walidacje wejścia ------------------------------------------------
        self._ensure_ffmpeg_available()
        paths = self._validate_inputs(wav_paths)
        output = Path(output_mp3).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)

        # 2. Plik z listą sąsiaduje z wyjściem, żeby był w tym samym mountcie.
        concat_list = output.parent / self.CONCAT_LIST_NAME
        self._write_concat_list(paths, concat_list)

        # 3. Uruchamiamy ffmpeg -----------------------------------------------
        try:
            self._run_ffmpeg(concat_list, output)
        except AudioStitcherError:
            # Świadomie NIE czyścimy plików po błędzie - są potrzebne do
            # diagnostyki przez użytkownika / dewelopera.
            logger.error(
                "FFmpeg zawiódł - pozostawiam %d plików .wav oraz listę "
                "concat na dysku dla diagnostyki.",
                len(paths),
            )
            raise

        if not output.exists() or output.stat().st_size == 0:
            raise AudioStitcherError(
                f"FFmpeg zakończył sukcesem, ale plik wyjściowy jest pusty: "
                f"{output}"
            )

        # 4. Sprzątanie po sukcesie -------------------------------------------
        cleanup_done = False
        if self._cleanup:
            self._cleanup_temp_files(paths, concat_list)
            cleanup_done = True

        return StitchResult(
            output_path=output,
            input_count=len(paths),
            output_size_bytes=output.stat().st_size,
            cleanup_performed=cleanup_done,
        )

    # ----- Wewnętrzne --------------------------------------------------------

    def _ensure_ffmpeg_available(self) -> None:
        if shutil.which(self._ffmpeg_bin) is None:
            raise AudioStitcherError(
                f"Nie znaleziono binarki '{self._ffmpeg_bin}' w PATH. "
                "Zainstaluj FFmpeg lub podaj pełną ścieżkę w konstruktorze."
            )

    @staticmethod
    def _validate_inputs(
        wav_paths: Sequence[str | Path],
    ) -> list[Path]:
        if not wav_paths:
            raise AudioStitcherError(
                "Pusta lista plików wejściowych - nie ma co łączyć."
            )
        resolved: list[Path] = []
        for p in wav_paths:
            path = Path(p).resolve()
            if not path.is_file():
                raise AudioStitcherError(
                    f"Plik wejściowy nie istnieje: {path}"
                )
            resolved.append(path)
        return resolved

    @staticmethod
    def _write_concat_list(paths: list[Path], list_path: Path) -> None:
        """
        Tworzy plik w formacie wymaganym przez `ffmpeg -f concat`.

        Apostrofy w ścieżkach wymagają escape'owania zgodnie z dokumentacją
        FFmpeg: zamykamy apostrof, dajemy '\\'' i otwieramy nowy apostrof.
        """
        with list_path.open("w", encoding="utf-8") as fh:
            for p in paths:
                safe_path = str(p).replace("'", r"'\''")
                fh.write(f"file '{safe_path}'\n")

    def _run_ffmpeg(self, concat_list: Path, output: Path) -> None:
        cmd: list[str] = [
            self._ffmpeg_bin,
            "-y",                       # nadpisuj istniejący output bez pytania
            "-hide_banner",
            "-loglevel", "error",
            "-f", "concat",
            "-safe", "0",               # zezwól na ścieżki bezwzględne / spec znaki
            "-i", str(concat_list),
            "-c:a", "libmp3lame",
            "-b:a", self._bitrate,
            "-ar", str(self._sample_rate),
            str(output),
        ]
        logger.info(
            "Łączę %s -> %s (bitrate=%s)",
            concat_list.name,
            output.name,
            self._bitrate,
        )
        logger.debug("FFmpeg cmd: %s", " ".join(cmd))

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            raise AudioStitcherError(
                f"FFmpeg zwrócił kod {exc.returncode}.\n"
                f"stderr:\n{exc.stderr}"
            ) from exc
        except FileNotFoundError as exc:
            # Race condition - shutil.which przepuściło, ale binarka znikła.
            raise AudioStitcherError(
                f"FFmpeg zniknął z systemu pomiędzy walidacją a uruchomieniem: {exc}"
            ) from exc

        if proc.stderr:
            # ffmpeg loguje także postęp na stderr - przekazujemy do loggera
            # na poziomie DEBUG, nie traktujemy jako błąd.
            logger.debug("FFmpeg stderr: %s", proc.stderr.strip())

    def _cleanup_temp_files(
        self, wav_paths: list[Path], concat_list: Path
    ) -> None:
        """
        Usuwa pliki .wav oraz plik listy. Błędy logujemy, ale nie rzucamy -
        główny artefakt (.mp3) już istnieje, więc sukces jest sukcesem.
        """
        removed = 0
        for p in wav_paths:
            try:
                p.unlink()
                removed += 1
            except OSError as exc:
                logger.warning("Nie udało się usunąć %s: %s", p, exc)

        try:
            concat_list.unlink()
        except OSError as exc:
            logger.warning("Nie udało się usunąć %s: %s", concat_list, exc)

        logger.info("Posprzątano %d plików .wav po sklejeniu.", removed)
