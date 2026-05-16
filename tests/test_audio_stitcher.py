"""
Testy jednostkowe dla AudioStitcher.

Strategia:
    * "Happy path" testujemy realnym FFmpeg-iem - mamy go w PATH i jest
      szybki dla krótkich WAV-ów wygenerowanych przez mock TTSModel.
    * Branżę błędów (FFmpeg zwraca niezerowy kod, brak binarki) symulujemy
      przez monkeypatch / subprocess mock.

Pokrywają:
    * walidacja wejścia: pusta lista, brak pliku, brak ffmpeg,
    * budowa pliku concat_list (escape apostrofu),
    * sukces: powstaje .mp3 z poprawnym MIME, ma sensowny rozmiar,
    * cleanup: po sukcesie .wav-y i lista znikają (z flagą cleanup=True),
    * cleanup wyłączony: pliki zostają,
    * porażka FFmpeg: pliki ZAWSZE zostają (do diagnostyki).
"""
from __future__ import annotations

import shutil
import subprocess
import wave
from pathlib import Path
from unittest.mock import patch

import pytest

from app.core.tts_model import SingletonMeta, TTSModel
from app.services.audio_stitcher import (
    AudioStitcher,
    AudioStitcherError,
    StitchResult,
)


# ============================================================
#  Fixture - świeży singleton TTS w mock_mode, plus N krótkich WAV-ów
# ============================================================


@pytest.fixture(autouse=True)
def _reset_tts_singleton():
    SingletonMeta._reset(TTSModel)
    yield
    SingletonMeta._reset(TTSModel)


@pytest.fixture
def wav_chunks(tmp_path: Path) -> list[Path]:
    """Tworzy 3 krótkie pliki .wav używając naszego mocka TTS."""
    model = TTSModel(mock_mode=True)
    paths: list[Path] = []
    for i, text in enumerate(
        ["Pierwszy chunk audio.", "Drugi chunk audio.", "Trzeci chunk."],
        start=1,
    ):
        p = tmp_path / f"chunk_{i:03d}.wav"
        model.synthesize_chunk(text, p)
        paths.append(p)
    return paths


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


# Skip cały moduł testów ffmpeg-zależnych, jeśli brakuje binarki.
needs_ffmpeg = pytest.mark.skipif(
    not _ffmpeg_available(),
    reason="FFmpeg nie jest zainstalowany w systemie testowym.",
)


# ============================================================
#  Walidacja wejścia
# ============================================================


class TestWalidacja:
    def test_pusta_lista_rzuca_blad(self, tmp_path: Path):
        stitcher = AudioStitcher()
        with pytest.raises(AudioStitcherError, match="Pusta lista"):
            stitcher.stitch([], tmp_path / "out.mp3")

    def test_nieistniejacy_plik_rzuca_blad(
        self, tmp_path: Path, wav_chunks: list[Path]
    ):
        stitcher = AudioStitcher()
        bad_list = [*wav_chunks, tmp_path / "nieistnieje.wav"]
        with pytest.raises(AudioStitcherError, match="nie istnieje"):
            stitcher.stitch(bad_list, tmp_path / "out.mp3")

    def test_brak_ffmpeg_rzuca_jasny_blad(
        self, tmp_path: Path, wav_chunks: list[Path]
    ):
        stitcher = AudioStitcher(ffmpeg_bin="ffmpeg_nie_istnieje_xyz")
        with pytest.raises(AudioStitcherError, match="Nie znaleziono"):
            stitcher.stitch(wav_chunks, tmp_path / "out.mp3")


# ============================================================
#  Budowa pliku concat (testowane bez wywoływania ffmpeg)
# ============================================================


class TestConcatList:
    def test_apostrofy_w_sciezce_sa_escape_owane(self, tmp_path: Path):
        # Tworzymy katalog z apostrofem w nazwie (jeśli system pozwoli).
        weird = tmp_path / "kat'alog"
        weird.mkdir()
        wav = weird / "audio.wav"
        wav.write_bytes(b"RIFF\x00\x00\x00\x00WAVE")  # atrapa - nie odpalamy ffmpeg

        list_file = tmp_path / "list.txt"
        AudioStitcher._write_concat_list([wav], list_file)

        content = list_file.read_text(encoding="utf-8")
        # Apostrof został rozłożony zgodnie z dokumentacją FFmpeg:
        # zamknij apostrof, '\'', otwórz nowy.
        assert r"'\''" in content
        assert content.startswith("file '")
        assert content.rstrip().endswith("'")

    def test_kolejnosc_plikow_jest_zachowana(self, tmp_path: Path):
        paths = []
        for i in range(5):
            p = tmp_path / f"chunk_{i:03d}.wav"
            p.write_bytes(b"x")
            paths.append(p)

        list_file = tmp_path / "list.txt"
        AudioStitcher._write_concat_list(paths, list_file)

        lines = list_file.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 5
        # Każda linia zawiera ścieżkę w kolejności wejścia
        for i, line in enumerate(lines):
            assert f"chunk_{i:03d}.wav" in line


# ============================================================
#  Happy path - prawdziwy FFmpeg
# ============================================================


@needs_ffmpeg
class TestSukcesRealFFmpeg:
    def test_laczenie_trzech_chunkow_tworzy_mp3(
        self, tmp_path: Path, wav_chunks: list[Path]
    ):
        stitcher = AudioStitcher(cleanup=False)  # zostawiamy WAV do dalszych asercji
        out = tmp_path / "audiobook.mp3"

        result = stitcher.stitch(wav_chunks, out)

        assert isinstance(result, StitchResult)
        assert result.output_path == out
        assert result.input_count == 3
        assert out.exists()
        assert out.stat().st_size > 500  # MP3 z 3 chunków = co najmniej kilka KB

    def test_mp3_ma_naglowek_id3_lub_mpeg_frame(
        self, tmp_path: Path, wav_chunks: list[Path]
    ):
        stitcher = AudioStitcher(cleanup=False)
        out = tmp_path / "audiobook.mp3"
        stitcher.stitch(wav_chunks, out)

        # Pierwsze bajty: albo ID3 tag ('ID3'), albo bezpośrednio MPEG sync
        # frame (0xFF 0xFB / 0xFF 0xFA / 0xFF 0xF3 / 0xFF 0xF2).
        head = out.read_bytes()[:4]
        is_id3 = head[:3] == b"ID3"
        is_mpeg = head[0] == 0xFF and (head[1] & 0xE0) == 0xE0
        assert is_id3 or is_mpeg, (
            f"Pierwsze bajty nie wyglądają na MP3: {head!r}"
        )


# ============================================================
#  Cleanup
# ============================================================


@needs_ffmpeg
class TestCleanup:
    def test_po_sukcesie_pliki_wav_sa_usuniete(
        self, tmp_path: Path, wav_chunks: list[Path]
    ):
        stitcher = AudioStitcher(cleanup=True)
        out = tmp_path / "audiobook.mp3"

        result = stitcher.stitch(wav_chunks, out)

        assert result.cleanup_performed is True
        for p in wav_chunks:
            assert not p.exists(), f"WAV powinien być usunięty: {p}"
        # Lista concat też usunięta
        assert not (out.parent / AudioStitcher.CONCAT_LIST_NAME).exists()
        # MP3 oczywiście zostaje
        assert out.exists()

    def test_cleanup_false_zostawia_wav_y(
        self, tmp_path: Path, wav_chunks: list[Path]
    ):
        stitcher = AudioStitcher(cleanup=False)
        out = tmp_path / "audiobook.mp3"

        result = stitcher.stitch(wav_chunks, out)

        assert result.cleanup_performed is False
        for p in wav_chunks:
            assert p.exists()

    def test_blad_ffmpeg_nie_kasuje_plikow(
        self, tmp_path: Path, wav_chunks: list[Path]
    ):
        """Po porażce FFmpeg .wav-y MUSZĄ zostać - są dowodem dla
        diagnostyki."""
        stitcher = AudioStitcher(cleanup=True)
        out = tmp_path / "audiobook.mp3"

        # Mockujemy subprocess.run tak, by zwracał niezerowy kod
        fake_error = subprocess.CalledProcessError(
            returncode=1,
            cmd=["ffmpeg"],
            stderr="Synthesized FFmpeg error for testing.",
        )
        with patch(
            "app.services.audio_stitcher.subprocess.run", side_effect=fake_error
        ):
            with pytest.raises(AudioStitcherError, match="kod 1"):
                stitcher.stitch(wav_chunks, out)

        # Wszystkie .wav-y nadal istnieją
        for p in wav_chunks:
            assert p.exists(), f"WAV nie powinien zniknąć po błędzie: {p}"
        # Lista concat też zostaje
        assert (out.parent / AudioStitcher.CONCAT_LIST_NAME).exists()


# ============================================================
#  StitchResult - dane wynikowe
# ============================================================


@needs_ffmpeg
class TestStitchResult:
    def test_pelne_dane_w_wyniku(
        self, tmp_path: Path, wav_chunks: list[Path]
    ):
        out = tmp_path / "audiobook.mp3"
        result = AudioStitcher(cleanup=False).stitch(wav_chunks, out)

        assert result.output_path == out
        assert result.input_count == len(wav_chunks)
        assert result.output_size_bytes == out.stat().st_size
        assert result.output_size_bytes > 0
        assert result.cleanup_performed is False


# ============================================================
#  Smoke test - krótki audiobook end-to-end (parser jest pominięty,
#  testujemy łańcuch TTS-mock -> stitcher)
# ============================================================


@needs_ffmpeg
class TestSmokeE2E:
    def test_pelny_pipeline_tts_mock_plus_stitcher(self, tmp_path: Path):
        """Symuluje: chunker dostarcza listę zdań -> TTS generuje WAV-y
        -> stitcher łączy w MP3."""
        chunks = [
            "Pierwszy rozdział mojej książki testowej.",
            "Drugie zdanie tego samego rozdziału, troszkę dłuższe.",
            "Trzeci fragment, podobnie krótki jak poprzedni.",
            "Czwarty i ostatni chunk - tu kończymy rozdział.",
        ]

        # 1) Synteza każdego chunka do osobnego .wav
        model = TTSModel(mock_mode=True)
        wav_paths: list[Path] = []
        for i, text in enumerate(chunks, start=1):
            wav = tmp_path / f"chunk_{i:03d}.wav"
            model.synthesize_chunk(text, wav)
            wav_paths.append(wav)

        # Weryfikacja pośrednia - WAV-y mają poprawne nagłówki
        for w in wav_paths:
            with wave.open(str(w), "rb") as f:
                assert f.getnchannels() == 1
                assert f.getframerate() == 24_000

        # 2) Stitcher tworzy MP3 i sprząta WAV-y
        result = AudioStitcher().stitch(wav_paths, tmp_path / "rozdzial_01.mp3")

        assert result.output_path.exists()
        assert result.cleanup_performed
        # WAV-y zniknęły, MP3 zostało
        assert all(not w.exists() for w in wav_paths)
        assert (tmp_path / "rozdzial_01.mp3").stat().st_size > 0
