"""
Testy jednostkowe dla TTSModel (singleton TTS).

Pokrywają:
    * wzorzec Singleton - dwa wywołania zwracają tę samą instancję,
    * thread-safety - 10 wątków tworzy 1 instancję,
    * lazy load i idempotentność load() / unload(),
    * walidacja wejścia (pusty tekst),
    * struktura wygenerowanego mock-WAV (header RIFF, sample rate, mono),
    * długość audio proporcjonalna do długości tekstu,
    * wywołanie torch.cuda.empty_cache() po każdej syntezie (mock-patched),
    * tryb produkcyjny rzuca jasny błąd bez speaker_wav.
"""
from __future__ import annotations

import threading
import wave
from pathlib import Path
from unittest.mock import patch

import pytest

from app.core.tts_model import (
    SingletonMeta,
    TTSModel,
    TTSModelError,
    _MockBackend,
)


# ============================================================
#  Fixture - reset singletona PRZED I PO każdym teście
# ============================================================


@pytest.fixture(autouse=True)
def _reset_tts_singleton():
    """Bez tego testy się wzajemnie zarażają instancjami."""
    SingletonMeta._reset(TTSModel)
    yield
    SingletonMeta._reset(TTSModel)


# ============================================================
#  Wzorzec Singleton
# ============================================================


class TestSingleton:
    def test_dwa_wywolania_zwracaja_ta_sama_instancje(self):
        a = TTSModel(mock_mode=True)
        b = TTSModel(mock_mode=True)
        assert a is b

    def test_argumenty_drugiego_wywolania_sa_ignorowane(self):
        a = TTSModel(mock_mode=True, language="pl")
        b = TTSModel(mock_mode=True, language="en")  # zignorowane
        assert a is b
        assert a._language == "pl"  # bez zmiany

    def test_reset_singletona_pozwala_utworzyc_nowa_instancje(self):
        a = TTSModel(mock_mode=True)
        SingletonMeta._reset(TTSModel)
        b = TTSModel(mock_mode=True)
        assert a is not b

    def test_thread_safe_tylko_jedna_instancja_pod_obciazeniem(self):
        instances: list[TTSModel] = []
        barrier = threading.Barrier(10)

        def worker():
            barrier.wait()  # synchronizacja - wszystkie 10 wątków startuje razem
            instances.append(TTSModel(mock_mode=True))

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Wszystkie wątki dostały TĘ SAMĄ instancję.
        assert len(instances) == 10
        assert all(inst is instances[0] for inst in instances)


# ============================================================
#  Cykl życia: load / unload
# ============================================================


class TestCyklZycia:
    def test_swiezo_utworzona_instancja_nie_jest_zaladowana(self):
        model = TTSModel(mock_mode=True)
        assert not model.is_loaded

    def test_load_laduje_mock_backend(self):
        model = TTSModel(mock_mode=True)
        model.load()
        assert model.is_loaded
        assert isinstance(model._model, _MockBackend)

    def test_load_jest_idempotentny(self):
        model = TTSModel(mock_mode=True)
        model.load()
        backend_id = id(model._model)
        model.load()  # drugie wywołanie - bez zmian
        model.load()  # trzecie - bez zmian
        assert id(model._model) == backend_id

    def test_unload_zwalnia_model(self):
        model = TTSModel(mock_mode=True)
        model.load()
        assert model.is_loaded
        model.unload()
        assert not model.is_loaded

    def test_unload_jest_idempotentny(self):
        model = TTSModel(mock_mode=True)
        model.unload()  # bez load() wcześniej - nie powinno wybuchnąć
        model.unload()  # drugie - też bez efektu

    def test_load_po_unload_dziala(self):
        model = TTSModel(mock_mode=True)
        model.load()
        model.unload()
        model.load()
        assert model.is_loaded


# ============================================================
#  Synteza - walidacja wejścia
# ============================================================


class TestWalidacjaWejscia:
    def test_pusty_tekst_rzuca_value_error(self, tmp_path: Path):
        model = TTSModel(mock_mode=True)
        with pytest.raises(ValueError, match="Pusty tekst"):
            model.synthesize_chunk("", tmp_path / "out.wav")

    def test_same_biale_znaki_rzucaja_value_error(self, tmp_path: Path):
        model = TTSModel(mock_mode=True)
        with pytest.raises(ValueError, match="Pusty tekst"):
            model.synthesize_chunk("   \n\t  ", tmp_path / "out.wav")

    def test_synteza_tworzy_brakujace_katalogi(self, tmp_path: Path):
        model = TTSModel(mock_mode=True)
        out = tmp_path / "deep" / "nested" / "chunk.wav"
        result = model.synthesize_chunk("Test polskiej mowy.", out)
        assert result.exists()
        assert result == out


# ============================================================
#  Synteza - poprawność pliku WAV
# ============================================================


class TestPlikiWAV:
    def test_synteza_tworzy_poprawny_plik_wav(self, tmp_path: Path):
        model = TTSModel(mock_mode=True)
        out = tmp_path / "chunk_001.wav"

        result = model.synthesize_chunk("Krótkie zdanie testowe.", out)

        assert result.exists()
        assert result.stat().st_size > 100  # nagłówek + jakieś sample

        # Czytamy nagłówek - musi być poprawny WAV
        with wave.open(str(result), "rb") as wav:
            assert wav.getnchannels() == 1               # mono
            assert wav.getsampwidth() == 2               # 16-bit
            assert wav.getframerate() == 24_000          # XTTS-v2 sample rate
            assert wav.getnframes() > 0

    def test_dluzszy_tekst_daje_dluzsze_audio(self, tmp_path: Path):
        model = TTSModel(mock_mode=True)
        short_out = tmp_path / "short.wav"
        long_out = tmp_path / "long.wav"

        model.synthesize_chunk("Krótkie.", short_out)
        model.synthesize_chunk("To jest znacznie dłuższy tekst, "
                              "z większą liczbą słów i sylab.", long_out)

        with wave.open(str(short_out), "rb") as s, wave.open(str(long_out), "rb") as l:
            assert l.getnframes() > s.getnframes()

    def test_kazdy_chunk_jest_zapisywany_oddzielnie(self, tmp_path: Path):
        """Singleton nie kumuluje audio - każde wywołanie to świeży plik."""
        model = TTSModel(mock_mode=True)

        paths = []
        for i in range(3):
            p = tmp_path / f"chunk_{i:03d}.wav"
            model.synthesize_chunk(f"Chunk numer {i}.", p)
            paths.append(p)

        # Każdy plik istnieje i ma sensowny rozmiar
        for p in paths:
            assert p.exists()
            assert p.stat().st_size > 100


# ============================================================
#  Zarządzanie pamięcią GPU
# ============================================================


class TestPamiecGPU:
    def test_empty_cache_wolane_po_kazdym_chunku(self, tmp_path: Path):
        """torch.cuda.empty_cache() wywoływane przez _free_gpu_cache po
        każdej syntezie - sprawdzamy przez mock.

        Patchujemy import torch wewnątrz _free_gpu_cache, symulując
        sytuację, w której torch JEST zainstalowany i ma dostępną CUDĘ.
        """
        import sys
        import types

        fake_torch = types.SimpleNamespace(
            cuda=types.SimpleNamespace(
                is_available=lambda: True,
                empty_cache=lambda: empty_cache_calls.append(1),
            )
        )
        empty_cache_calls: list[int] = []

        with patch.dict(sys.modules, {"torch": fake_torch}):
            model = TTSModel(mock_mode=True)
            model.synthesize_chunk("Pierwszy chunk testowy.", tmp_path / "a.wav")
            model.synthesize_chunk("Drugi chunk testowy.", tmp_path / "b.wav")
            model.synthesize_chunk("Trzeci chunk testowy.", tmp_path / "c.wav")

        # Po 3 syntezach mamy 3 wywołania empty_cache (oraz ew. jedno z unload,
        # którego tu nie robimy).
        assert len(empty_cache_calls) >= 3

    def test_brak_torch_nie_psuje_syntezy(self, tmp_path: Path):
        """W trybie mock bez torch synteza musi działać bez błędów."""
        import sys

        # Wymuś brak torch w sys.modules
        with patch.dict(sys.modules, {"torch": None}):
            model = TTSModel(mock_mode=True)
            # Nie powinno rzucić ImportError, mimo że _free_gpu_cache
            # próbuje zaimportować torch.
            result = model.synthesize_chunk("Test bez torch.", tmp_path / "x.wav")
            assert result.exists()


# ============================================================
#  Tryb produkcyjny - obecnie wyłączony, ale stub musi działać sensownie
# ============================================================


class TestTrybProdukcyjny:
    def test_brak_speaker_wav_rzuca_jasny_blad(self, tmp_path: Path):
        model = TTSModel(mock_mode=False, speaker_wav=None)
        with pytest.raises(TTSModelError, match="speaker_wav"):
            model.load()

    def test_nieistniejacy_speaker_wav_rzuca_jasny_blad(self, tmp_path: Path):
        model = TTSModel(
            mock_mode=False, speaker_wav=tmp_path / "brak.wav"
        )
        with pytest.raises(TTSModelError, match="referencyjnego"):
            model.load()

    def test_load_real_jest_jeszcze_stubem(self, tmp_path: Path):
        """Jak długo nie odkomentujemy produkcyjnego kodu, load() rzuca
        sensowny błąd z instrukcją co zrobić."""
        speaker = tmp_path / "speaker.wav"
        speaker.write_bytes(b"fake-wav-bytes")
        model = TTSModel(mock_mode=False, speaker_wav=speaker)
        with pytest.raises(TTSModelError, match="Odkomentuj"):
            model.load()
