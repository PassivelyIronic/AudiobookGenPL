"""
TTS Model Singleton — zarządza pojedynczą instancją modelu Text-to-Speech
w cyklu życia procesu workera Celery.

Założenia projektowe:
    * Model XTTS-v2 zajmuje ~2 GB VRAM. RTX 4060 ma 8 GB, więc musimy się
      pilnować i NIGDY nie ładować modelu dwa razy w tym samym procesie.
    * Pojedynczy worker Celery = pojedynczy proces = pojedyncza instancja
      singletona. Multiprocessing-worker Celery utworzy oddzielne procesy,
      każdy ze swoją instancją - to oczekiwane zachowanie (jeden GPU,
      jeden worker dla TTS).
    * Synteza musi zwalniać tymczasowe tensory po każdym chunku
      (`torch.cuda.empty_cache()`), żeby długie książki nie nadmuchały
      VRAM przez fragmentację.

Tryb mockowy:
    * Domyślnie singleton działa w trybie `mock_mode=True`, w którym zamiast
      ładować PyTorch / TTS, generuje krótkie pliki .wav z sinusem.
      Dzięki temu cały pipeline (parser → chunker → TTS → stitcher) można
      testować bez GPU.
    * Tryb produkcyjny (`mock_mode=False`) ma już szkielet pod XTTS-v2
      z Coqui, ale wymaga doinstalowania `TTS` i `torch` (zakomentowane
      w requirements.txt).
"""
from __future__ import annotations

import logging
import math
import struct
import threading
import wave
from pathlib import Path
from typing import Any, ClassVar, Final

logger = logging.getLogger(__name__)


# ============================================================
#  Wyjątki
# ============================================================


class TTSModelError(Exception):
    """Błąd ładowania modelu lub syntezy mowy."""


# ============================================================
#  Metaklasa Singleton (thread-safe)
# ============================================================


class SingletonMeta(type):
    """
    Metaklasa wymuszająca, że dla danej klasy istnieje co najwyżej jedna
    instancja w procesie.

    Dwa wywołania `TTSModel()` zwrócą TĘ SAMĄ instancję. Argumenty drugiego
    wywołania są IGNOROWANE (instancja już istnieje) - logujemy ostrzeżenie,
    jeśli różnią się od pierwszego.
    """

    _instances: ClassVar[dict[type, Any]] = {}
    _lock: ClassVar[threading.Lock] = threading.Lock()

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        # Podwójne sprawdzenie (double-checked locking) - typowy pattern,
        # żeby nie blokować na każdy odczyt po pierwszej inicjalizacji.
        if cls not in cls._instances:
            with cls._lock:
                if cls not in cls._instances:
                    cls._instances[cls] = super().__call__(*args, **kwargs)
                    return cls._instances[cls]

        if args or kwargs:
            logger.warning(
                "Singleton %s już istnieje - argumenty %s/%s zignorowane.",
                cls.__name__,
                args,
                kwargs,
            )
        return cls._instances[cls]

    @classmethod
    def _reset(mcs, target_cls: type | None = None) -> None:
        """
        WYŁĄCZNIE DO TESTÓW: resetuje rejestr singletonów.

        Nie używać w kodzie produkcyjnym - jeśli czujesz, że musisz,
        to znaczy, że gdzieś masz błąd projektowy.
        """
        with mcs._lock:
            if target_cls is None:
                mcs._instances.clear()
            else:
                mcs._instances.pop(target_cls, None)


# ============================================================
#  Mock backend (dla testów i developmentu bez GPU)
# ============================================================


class _MockBackend:
    """
    Generator atrap plików .wav - krótka sinusoida 440 Hz proporcjonalna
    do długości tekstu. Pliki są PRAWDZIWE - mają poprawny nagłówek WAV
    i można je odtworzyć w odtwarzaczu lub przekazać do FFmpeg.
    """

    SAMPLE_RATE: Final[int] = 24_000  # matchuje XTTS-v2
    AMPLITUDE: Final[int] = 8_000     # ~25% maks dla 16-bit (-32768..32767)
    FREQ_HZ: Final[float] = 440.0     # nuta A4

    # ~10 znaków/sek - proporcja podobna do realnej polskiej mowy.
    CHARS_PER_SECOND: Final[float] = 10.0
    MIN_DURATION_SEC: Final[float] = 0.3

    def synthesize(self, text: str, output_path: Path) -> None:
        duration = max(self.MIN_DURATION_SEC, len(text) / self.CHARS_PER_SECOND)
        n_samples = int(self.SAMPLE_RATE * duration)

        # Generujemy w pamięci tylko jeden chunk - dla audiobooka z reguły
        # < 1 MB na chunk, więc pamięć nie jest tu problemem.
        samples = bytearray()
        for i in range(n_samples):
            value = int(
                self.AMPLITUDE
                * math.sin(2 * math.pi * self.FREQ_HZ * i / self.SAMPLE_RATE)
            )
            samples.extend(struct.pack("<h", value))

        with wave.open(str(output_path), "wb") as wav:
            wav.setnchannels(1)        # mono
            wav.setsampwidth(2)        # 16-bit
            wav.setframerate(self.SAMPLE_RATE)
            wav.writeframes(bytes(samples))


# ============================================================
#  Singleton TTSModel
# ============================================================


class TTSModel(metaclass=SingletonMeta):
    """
    Singleton ładujący model TTS raz na cykl życia procesu workera.

    Użycie w Celery:
        # przy starcie workera (signal worker_process_init):
        TTSModel(mock_mode=False).load()

        # w tasku:
        TTSModel().synthesize_chunk("Tekst do mowy.", Path("chunk_001.wav"))

    Wszystkie kolejne wywołania `TTSModel()` zwrócą tę samą instancję.
    """

    DEFAULT_MODEL_NAME: Final[str] = "tts_models/multilingual/multi-dataset/xtts_v2"
    DEFAULT_LANGUAGE: Final[str] = "pl"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        language: str = DEFAULT_LANGUAGE,
        speaker_wav: str | Path | None = None,
        device: str = "cuda",
        mock_mode: bool = True,
    ) -> None:
        """
        Args:
            model_name: identyfikator modelu w katalogu Coqui (tylko gdy
                mock_mode=False).
            language: kod języka (XTTS-v2 wspiera 'pl').
            speaker_wav: ścieżka do nagrania głosu referencyjnego dla voice
                cloningu (XTTS-v2). Wymagane w trybie produkcyjnym.
            device: 'cuda' albo 'cpu'.
            mock_mode: jeśli True (domyślnie), generuje atrapy WAV zamiast
                ładować model. Idealne do dev / testów / CI.
        """
        self._model_name: str = model_name
        self._language: str = language
        self._speaker_wav: Path | None = (
            Path(speaker_wav) if speaker_wav else None
        )
        self._device: str = device
        self._mock_mode: bool = mock_mode

        self._model: Any = None
        self._lock: threading.Lock = threading.Lock()

        logger.info(
            "TTSModel zainicjalizowany (mock_mode=%s, device=%s, language=%s)",
            self._mock_mode,
            self._device,
            self._language,
        )

    # ----- Cykl życia modelu -------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def mock_mode(self) -> bool:
        return self._mock_mode

    def load(self) -> None:
        """
        Ładuje model do VRAM. Idempotentne - bezpiecznie wywoływane wielokrotnie.

        Wywoływać przy starcie workera Celery (signal worker_process_init),
        żeby pierwszy task nie czekał na 10-15 sek ładowania modelu.
        """
        with self._lock:
            if self._model is not None:
                logger.debug("TTSModel.load(): model już załadowany, pomijam.")
                return

            if self._mock_mode:
                logger.info("Ładuję mock backend TTS (bez GPU).")
                self._model = _MockBackend()
                return

            self._model = self._load_real_model()

    def unload(self) -> None:
        """
        Zwalnia model z VRAM. Przydatne, gdy worker przechodzi w stan idle
        lub shutdown. Idempotentne.
        """
        with self._lock:
            if self._model is None:
                return
            logger.info("Zwalniam model TTS z pamięci.")
            self._model = None
            self._free_gpu_cache()

    # ----- Synteza -----------------------------------------------------------

    def synthesize_chunk(self, text: str, output_path: str | Path) -> Path:
        """
        Generuje plik .wav dla pojedynczego chunka tekstu i zapisuje go
        na dysk - NIGDY nie zwraca audio w pamięci.

        Args:
            text: chunk tekstu (max ~500 znaków, zgodnie z chunkerem).
            output_path: docelowa ścieżka pliku .wav.

        Returns:
            Path: ścieżka do utworzonego pliku.

        Raises:
            TTSModelError: gdy synteza zawiedzie.
            ValueError: gdy `text` jest pusty.
        """
        if not text or not text.strip():
            raise ValueError("Pusty tekst - nie ma co syntezować.")

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        # Lazy load - na wypadek, gdyby ktoś zapomniał wywołać load() przy
        # starcie workera.
        if not self.is_loaded:
            self.load()

        try:
            if self._mock_mode:
                self._model.synthesize(text, out)
            else:
                self._synthesize_real(text, out)
        except Exception as exc:  # noqa: BLE001
            raise TTSModelError(
                f"Synteza nie powiodła się dla tekstu {text[:50]!r}: {exc}"
            ) from exc
        finally:
            # KRYTYCZNE: zwalniamy cache CUDA po każdym chunku, żeby VRAM
            # nie nadymał się przy długich książkach.
            self._free_gpu_cache()

        if not out.exists() or out.stat().st_size == 0:
            raise TTSModelError(
                f"Synteza zakończona, ale plik nie powstał: {out}"
            )

        return out

    # ----- Implementacje prawdziwego modelu (do wypełnienia w fazie TTS) -----

    def _load_real_model(self) -> Any:
        """
        Ładuje XTTS-v2 z forka coqui-tts (idiap/coqui-ai-TTS).

        Pierwsze wywołanie pobiera model z HuggingFace (~1.8 GB) do cache:
            ~/.local/share/tts/                  (Linux)
            %LOCALAPPDATA%\\tts\\                  (Windows)
        Kolejne starty workera ładują z cache - zajmuje 10-20 sek + ~3 GB VRAM.
        """
        if self._speaker_wav is None:
            raise TTSModelError(
                "speaker_wav jest wymagany dla XTTS-v2 (voice cloning)."
            )
        if not self._speaker_wav.is_file():
            raise TTSModelError(
                f"Plik głosu referencyjnego nie istnieje: {self._speaker_wav}"
            )

        # Importy lazy - tylko gdy faktycznie ładujemy prawdziwy model.
        # Dzięki temu tryb mock działa bez instalacji torcha / coqui-tts.
        from TTS.api import TTS  # noqa: PLC0415
        import torch  # noqa: PLC0415

        if self._device == "cuda" and not torch.cuda.is_available():
            logger.warning(
                "CUDA niedostępna, przełączam na CPU - synteza będzie 10-50x wolniejsza."
            )
            self._device = "cpu"

        logger.info(
            "Ładuję XTTS-v2 (%s) na %s ...", self._model_name, self._device
        )
        tts = TTS(self._model_name).to(self._device)
        logger.info(
            "Model TTS załadowany. VRAM zajęty: %.2f GB",
            torch.cuda.memory_allocated() / 1024**3 if self._device == "cuda" else 0.0,
        )
        return tts

    def _synthesize_real(self, text: str, output_path: Path) -> None:
        """Synteza w trybie produkcyjnym XTTS-v2 z voice cloningiem."""
        # split_sentences=True: XTTS sam decyduje, jak podzielić wewnątrz chunka.
        # Nasz chunker jest "ceiling" rozmiaru wsadu, XTTS jest sprytniejszy
        # w segmentacji audio dla naturalnej prozodii.
        self._model.tts_to_file(
            text=text,
            speaker_wav=str(self._speaker_wav),
            language=self._language,
            file_path=str(output_path),
            split_sentences=True,
        )

    # ----- Zarządzanie pamięcią GPU ------------------------------------------

    @staticmethod
    def _free_gpu_cache() -> None:
        """
        Zwalnia cache CUDA, jeśli torch jest dostępny. Cicho ignoruje brak.
        """
        try:
            import torch  # noqa: PLC0415 - import warunkowy

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            # torch nie jest zainstalowany - tryb mock / CPU
            pass
