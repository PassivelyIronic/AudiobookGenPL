"""
Testy jednostkowe dla EpubParser.

Każdy test buduje minimalny, ale poprawny .epub w tmp_path zamiast
korzystać z plików zewnętrznych. Dzięki temu testy są deterministyczne
i nie wymagają fixture'ów na dysku.

Pokrywają:
    * walidację ścieżki (nieistniejący plik, zły suffix),
    * wycinanie tagów <script>/<style>/<nav>/<header>/<footer>,
    * zachowanie polskich znaków diakrytycznych,
    * normalizację białych znaków,
    * strumieniowy charakter iter_chapters (rozdziały yieldowane lazy).
"""
from __future__ import annotations

from pathlib import Path

import pytest
from ebooklib import epub

from app.services.epub_parser import Chapter, EpubParser, EpubParserError


# ============================================================
#  Pomocnicze - budowanie minimalnego EPUB
# ============================================================


def _build_minimal_epub(tmp_path: Path, chapter_htmls: list[str]) -> Path:
    """Zwraca ścieżkę do świeżo zbudowanego, poprawnego pliku .epub."""
    book = epub.EpubBook()
    book.set_identifier("test-audio-ksiaznica")
    book.set_title("Książka testowa")
    book.set_language("pl")
    book.add_author("Zespół Testów")

    items: list[epub.EpubHtml] = []
    for i, body_html in enumerate(chapter_htmls, start=1):
        chapter = epub.EpubHtml(
            title=f"Rozdział {i}",
            file_name=f"chap_{i}.xhtml",
            lang="pl",
        )
        # WAŻNE: musi być bytes - jeśli przekażemy str do .content,
        # ebooklib przy zapisie EPUB gubi treść (zostaje pusty plik).
        chapter.set_content(
            (
                "<?xml version='1.0' encoding='utf-8'?>"
                "<html xmlns='http://www.w3.org/1999/xhtml'>"
                "<head><title>Test</title></head>"
                f"<body>{body_html}</body>"
                "</html>"
            ).encode("utf-8")
        )
        book.add_item(chapter)
        items.append(chapter)

    book.toc = tuple(items)
    book.add_item(epub.EpubNcx())
    # UWAGA: NIE dodajemy epub.EpubNav() - bez ustawionego content
    # ebooklib rzuca lxml.etree.ParserError: Document is empty.
    # Do naszych testów (czytanie ITEM_DOCUMENT) NAV nie jest potrzebny.
    book.spine = list(items)

    out = tmp_path / "test_book.epub"
    epub.write_epub(str(out), book, {})
    return out


def _make_parser(tmp_path: Path, html_body: str) -> EpubParser:
    """Skrót: jeden rozdział z podanym body HTML."""
    return EpubParser(_build_minimal_epub(tmp_path, [html_body]))


# ============================================================
#  Walidacja inicjalizacji
# ============================================================


class TestInicjalizacja:
    def test_brak_pliku_rzuca_wyjatek(self):
        with pytest.raises(EpubParserError, match="nie istnieje"):
            EpubParser("/nieistniejacy/folder/ksiazka.epub")

    def test_zle_rozszerzenie_rzuca_wyjatek(self, tmp_path: Path):
        plik = tmp_path / "ksiazka.txt"
        plik.write_text("treść", encoding="utf-8")
        with pytest.raises(EpubParserError, match="rozszerzenia"):
            EpubParser(plik)

    def test_uszkodzony_epub_rzuca_wyjatek(self, tmp_path: Path):
        plik = tmp_path / "uszkodzony.epub"
        plik.write_bytes(b"to nie jest poprawny epub")
        parser = EpubParser(plik)
        with pytest.raises(EpubParserError):
            list(parser.iter_chapters())


# ============================================================
#  Czyszczenie HTML
# ============================================================


class TestCzyszczenieHTML:
    def test_wycina_tagi_script(self, tmp_path: Path):
        parser = _make_parser(
            tmp_path,
            "<p>Treść rozdziału testowego pierwsza linia.</p>"
            "<script>alert('zlosliwy kod')</script>"
            "<p>Treść kolejna druga linia tekstu.</p>",
        )
        text = list(parser.iter_chapters())[0].text
        assert "alert" not in text
        assert "zlosliwy" not in text
        assert "Treść rozdziału" in text
        assert "Treść kolejna" in text

    def test_wycina_tagi_style(self, tmp_path: Path):
        parser = _make_parser(
            tmp_path,
            "<style>.klasa { color: red; }</style>"
            "<p>Akapit z odpowiednią ilością treści do testu długości progu min_chars.</p>",
        )
        text = list(parser.iter_chapters())[0].text
        assert "color" not in text
        assert "klasa" not in text
        assert "Akapit" in text

    def test_wycina_naglowki_i_stopki_nawigacyjne(self, tmp_path: Path):
        parser = _make_parser(
            tmp_path,
            "<header>Nawigacja górna do usunięcia</header>"
            "<p>Właściwa treść rozdziału z odpowiednią ilością tekstu do testu progu.</p>"
            "<footer>Stopka prawnoautorska do usunięcia</footer>",
        )
        text = list(parser.iter_chapters())[0].text
        assert "Nawigacja górna" not in text
        assert "Stopka prawnoautorska" not in text
        assert "Właściwa treść" in text

    def test_brak_tagow_HTML_w_wyjsciu(self, tmp_path: Path):
        parser = _make_parser(
            tmp_path,
            "<p>Akapit <strong>z pogrubieniem</strong> i "
            "<em>kursywą</em> dla testu długości tekstu.</p>",
        )
        text = list(parser.iter_chapters())[0].text
        assert "<" not in text
        assert ">" not in text
        # Treść tekstowa zachowana
        assert "pogrubieniem" in text
        assert "kursywą" in text


# ============================================================
#  Polskie znaki
# ============================================================


class TestPolskieZnaki:
    def test_zachowanie_polskich_diakrytykow(self, tmp_path: Path):
        parser = _make_parser(
            tmp_path,
            "<p>Zażółć gęślą jaźń ŻÓŁWIA. "
            "Łódź jest miastem przyjaznym dla rowerzystów.</p>",
        )
        text = list(parser.iter_chapters())[0].text
        for char in "ąćęłńóśźżĄĆĘŁŃÓŚŹŻ":
            if char in "Zażółć gęślą jaźń ŻÓŁWIA Łódź":
                assert char in text, f"Brak polskiego znaku: {char}"


# ============================================================
#  Normalizacja białych znaków
# ============================================================


class TestBialeZnaki:
    def test_wielokrotne_spacje_sa_normalizowane(self, tmp_path: Path):
        parser = _make_parser(
            tmp_path,
            "<p>Tekst   z      wieloma\n\n\nspacjami\ti\ttabulacjami "
            "wystarczająco długi, by przeszedł próg min_chars.</p>",
        )
        text = list(parser.iter_chapters())[0].text
        assert "  " not in text  # brak podwójnych spacji
        assert "\n" not in text
        assert "\t" not in text

    def test_brak_spacji_przed_interpunkcja(self, tmp_path: Path):
        parser = _make_parser(
            tmp_path,
            "<p>Zdanie z interpunkcją . I drugie zdanie ! A także trzecie ? "
            "Czwarte na koniec , dla testu długości.</p>",
        )
        text = list(parser.iter_chapters())[0].text
        # Po normalizacji nie powinno być " ." " !" " ?" " ,"
        assert " ." not in text
        assert " !" not in text
        assert " ?" not in text
        assert " ," not in text


# ============================================================
#  Strumieniowanie rozdziałów
# ============================================================


class TestStrumieniowanie:
    def test_iter_chapters_zwraca_generator(self, tmp_path: Path):
        import types

        parser = EpubParser(
            _build_minimal_epub(
                tmp_path,
                ["<p>Rozdział pierwszy z odpowiednio długą treścią testową.</p>"],
            )
        )
        result = parser.iter_chapters()
        assert isinstance(result, types.GeneratorType)

    def test_kolejne_rozdzialy_maja_rosnacy_indeks(self, tmp_path: Path):
        parser = EpubParser(
            _build_minimal_epub(
                tmp_path,
                [
                    "<h1>Rozdział 1</h1><p>Treść pierwsza " + "lorem " * 20 + "</p>",
                    "<h1>Rozdział 2</h1><p>Treść druga " + "ipsum " * 20 + "</p>",
                    "<h1>Rozdział 3</h1><p>Treść trzecia " + "dolor " * 20 + "</p>",
                ],
            )
        )
        chapters = list(parser.iter_chapters())
        assert len(chapters) == 3
        assert [c.index for c in chapters] == [0, 1, 2]

    def test_krotkie_rozdzialy_sa_pomijane(self, tmp_path: Path):
        parser = EpubParser(
            _build_minimal_epub(
                tmp_path,
                [
                    "<p>Krótko.</p>",  # poniżej min_chars
                    "<p>Pełnoprawny rozdział z odpowiednią ilością treści, "
                    "który zdecydowanie powinien przejść przez próg długości.</p>",
                ],
            )
        )
        chapters = list(parser.iter_chapters(min_chars=50))
        assert len(chapters) == 1
        assert "Pełnoprawny rozdział" in chapters[0].text

    def test_tytul_z_h1_jest_uzywany(self, tmp_path: Path):
        parser = _make_parser(
            tmp_path,
            "<h1>Mój własny tytuł rozdziału</h1>"
            "<p>Treść rozdziału z wystarczającą długością do przejścia progu.</p>",
        )
        chapter = list(parser.iter_chapters())[0]
        assert chapter.title == "Mój własny tytuł rozdziału"

    def test_chapter_to_zamrozona_dataklasa(self, tmp_path: Path):
        parser = _make_parser(
            tmp_path,
            "<p>Treść rozdziału z wystarczającą długością do progu min_chars.</p>",
        )
        chapter = list(parser.iter_chapters())[0]
        assert isinstance(chapter, Chapter)
        with pytest.raises(Exception):  # frozen dataclass = FrozenInstanceError
            chapter.text = "podmiana"  # type: ignore[misc]
