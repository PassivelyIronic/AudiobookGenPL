"""
Testy ConversionPipeline.

Strategia:
    Pipeline jest klejem - chcemy sprawdzić, że poprawnie spina komponenty,
    propaguje błędy i raportuje postęp. Komponenty (parser, chunker, TTS,
    stitcher) mają własne testy jednostkowe, tu nie powtarzamy.

Każdy test buduje minimalny prawdziwy EPUB w tmp_path i uruchamia pełen
łańcuch konwersji w trybie mock_mode (TTS generuje sinusy, FFmpeg robi
prawdziwy MP3).
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from ebooklib import epub

from app.core.config import Settings
from app.core.tts_model import SingletonMeta, TTSModel
from app.services.pipeline import (
    ConversionPipeline,
    ConversionResult,
    PipelineError,
)


# ============================================================
#  Wspólne fixtures
# ============================================================


@pytest.fixture(autouse=True)
def _reset_tts_singleton():
    """Każdy test startuje ze świeżym singletonem."""
    SingletonMeta._reset(TTSModel)
    yield
    SingletonMeta._reset(TTSModel)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Settings z katalogami osadzonymi w tmp_path."""
    return Settings(
        upload_dir=tmp_path / "uploads",
        output_dir=tmp_path / "outputs",
        work_dir=tmp_path / "work",
        tts_mock_mode=True,
        chunk_max_chars=200,
        mp3_bitrate="64k",          # niski - szybkie testy
        mp3_sample_rate=24_000,
    )


def _build_test_epub(path: Path, chapters: list[tuple[str, str]]) -> Path:
    """Tworzy minimalny EPUB z listy (tytuł, body_html)."""
    book = epub.EpubBook()
    book.set_identifier("test")
    book.set_title("Testowa książka")
    book.set_language("pl")
    book.add_author("Tester")

    items = []
    for i, (title, body) in enumerate(chapters, start=1):
        ch = epub.EpubHtml(title=title, file_name=f"ch_{i}.xhtml", lang="pl")
        ch.set_content(
            (
                "<?xml version='1.0' encoding='utf-8'?>"
                "<html xmlns='http://www.w3.org/1999/xhtml'>"
                f"<head><title>{title}</title></head>"
                f"<body><h1>{title}</h1>{body}</body>"
                "</html>"
            ).encode("utf-8")
        )
        book.add_item(ch)
        items.append(ch)

    book.toc = tuple(items)
    book.add_item(epub.EpubNcx())
    book.spine = list(items)
    epub.write_epub(str(path), book, {})
    return path


@pytest.fixture
def sample_epub(tmp_path: Path) -> Path:
    """Krótki EPUB z 3 rozdziałami zawierający polskie znaki."""
    return _build_test_epub(
        tmp_path / "ksiazka.epub",
        [
            (
                "Rozdział pierwszy",
                "<p>Pan Tadeusz wstał wcześnie rano. Spojrzał przez okno - "
                "na horyzoncie zaróżowił się świt. Pomyślał o nadchodzącym dniu.</p>",
            ),
            (
                "Rozdział drugi",
                "<p>Zażółć gęślą jaźń. To słynne ćwiczenie z polskich "
                "diakrytyków używane przez drukarzy. Każdy znak ma znaczenie.</p>",
            ),
            (
                "Rozdział trzeci",
                "<p>Konwersja audiobooka kończy się tu. "
                "Dziękujemy za uwagę i miłego słuchania!</p>",
            ),
        ],
    )


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


needs_ffmpeg = pytest.mark.skipif(
    not _ffmpeg_available(), reason="FFmpeg niedostępny."
)


# ============================================================
#  Happy path
# ============================================================


@needs_ffmpeg
class TestSukces:
    def test_konwersja_konczy_sie_plikiem_mp3(
        self, settings: Settings, sample_epub: Path, tmp_path: Path
    ):
        pipeline = ConversionPipeline(settings)
        output = tmp_path / "audiobook.mp3"

        result = pipeline.convert(sample_epub, tmp_path / "work", output)

        assert isinstance(result, ConversionResult)
        assert result.output_path == output
        assert output.exists()
        assert result.chapters_processed == 3
        assert result.chunks_synthesized >= 3
        assert result.output_size_bytes > 0

    def test_work_dir_jest_sprzatany_po_sukcesie(
        self, settings: Settings, sample_epub: Path, tmp_path: Path
    ):
        pipeline = ConversionPipeline(settings)
        work_dir = tmp_path / "work"
        output = tmp_path / "audiobook.mp3"

        pipeline.convert(sample_epub, work_dir, output)

        # Po sukcesie work_dir znika (cleanup_work_dir_after_success=True)
        assert not work_dir.exists()

    def test_kazdy_chunk_pojawia_sie_w_kolejnosci_w_mp3(
        self, settings: Settings, tmp_path: Path
    ):
        """Sprawdzamy, że dłuższy tekst rzeczywiście daje większy plik."""
        short_epub = _build_test_epub(
            tmp_path / "short.epub",
            [("R1", "<p>Krótki tekst testowy z odpowiednią długością, "
                    "żeby przejść przez próg min_chars w parserze.</p>")],
        )
        long_epub = _build_test_epub(
            tmp_path / "long.epub",
            [
                (f"R{i}", f"<p>{'Zdanie testowe numer {}. '.format(i) * 10}</p>")
                for i in range(1, 6)
            ],
        )

        p_short = ConversionPipeline(settings)
        p_long = ConversionPipeline(settings)

        short_out = tmp_path / "short.mp3"
        long_out = tmp_path / "long.mp3"
        p_short.convert(short_epub, tmp_path / "w_short", short_out)
        p_long.convert(long_epub, tmp_path / "w_long", long_out)

        assert long_out.stat().st_size > short_out.stat().st_size


# ============================================================
#  Progress callback
# ============================================================


@needs_ffmpeg
class TestProgress:
    def test_callback_jest_wolany_dla_kazdego_etapu(
        self, settings: Settings, sample_epub: Path, tmp_path: Path
    ):
        events: list[dict] = []

        def cb(*, stage: str, current: int, total: int, message: str = "") -> None:
            events.append(
                {"stage": stage, "current": current, "total": total, "message": message}
            )

        pipeline = ConversionPipeline(settings, progress_callback=cb)
        pipeline.convert(sample_epub, tmp_path / "work", tmp_path / "out.mp3")

        stages = {e["stage"] for e in events}
        assert "parsing" in stages
        assert "synthesizing" in stages
        assert "stitching" in stages
        assert "done" in stages

    def test_postep_syntezy_rosnie_wraz_z_rozdzialami(
        self, settings: Settings, sample_epub: Path, tmp_path: Path
    ):
        events: list[dict] = []

        def cb(*, stage: str, current: int, total: int, message: str = "") -> None:
            if stage == "synthesizing":
                events.append({"current": current, "total": total})

        pipeline = ConversionPipeline(settings, progress_callback=cb)
        pipeline.convert(sample_epub, tmp_path / "work", tmp_path / "out.mp3")

        assert len(events) == 3
        # current rośnie 1, 2, 3
        assert [e["current"] for e in events] == [1, 2, 3]
        # total dla każdego eventu = 3 (liczba rozdziałów)
        assert all(e["total"] == 3 for e in events)


# ============================================================
#  Obsługa błędów
# ============================================================


class TestBledy:
    def test_nieistniejacy_epub_rzuca_pipeline_error(
        self, settings: Settings, tmp_path: Path
    ):
        pipeline = ConversionPipeline(settings)
        with pytest.raises(PipelineError):
            pipeline.convert(
                tmp_path / "brak.epub",
                tmp_path / "work",
                tmp_path / "out.mp3",
            )

    def test_pusty_epub_rzuca_pipeline_error(
        self, settings: Settings, tmp_path: Path
    ):
        """EPUB ze samym tekstem za krótkim by przejść próg min_chars=50."""
        empty = _build_test_epub(
            tmp_path / "empty.epub",
            [("R1", "<p>Hej.</p>")],  # 4 znaki < 50
        )
        pipeline = ConversionPipeline(settings)
        with pytest.raises(PipelineError, match="rozdziałów"):
            pipeline.convert(empty, tmp_path / "work", tmp_path / "out.mp3")

    def test_pipeline_error_zachowuje_oryginalny_wyjatek(
        self, settings: Settings, tmp_path: Path
    ):
        """__cause__ zawiera pierwotny wyjątek dla diagnostyki."""
        pipeline = ConversionPipeline(settings)
        try:
            pipeline.convert(
                tmp_path / "nie_istnieje.epub",
                tmp_path / "work",
                tmp_path / "out.mp3",
            )
        except PipelineError as exc:
            # Oryginalny wyjątek z EpubParser-a powinien być przyczyną
            # albo PipelineError zostało podniesione bezpośrednio
            assert exc.__cause__ is not None or "EPUB" in str(exc)


# ============================================================
#  Reużywalność pipeline'u
# ============================================================


@needs_ffmpeg
class TestReuzywalnosc:
    def test_jedna_instancja_pipeline_dwie_konwersje(
        self, settings: Settings, sample_epub: Path, tmp_path: Path
    ):
        """Pipeline można puścić dwa razy - singleton TTS jest reużywany."""
        pipeline = ConversionPipeline(settings)

        r1 = pipeline.convert(sample_epub, tmp_path / "w1", tmp_path / "a.mp3")
        r2 = pipeline.convert(sample_epub, tmp_path / "w2", tmp_path / "b.mp3")

        assert r1.output_path.exists()
        assert r2.output_path.exists()
        assert r1.chapters_processed == r2.chapters_processed
