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

from app.services.epub_parser import (
    Chapter,
    ChapterFilter,
    EpubParser,
    EpubParserError,
)


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


# ============================================================
#  Smart Parser - heurystyka pomijania śmieciowych rozdziałów
# ============================================================


class TestChapterFilterWhitelist:
    """Numerowane rozdziały - whitelist wygrywa nad wszystkim innym."""

    @pytest.mark.parametrize(
        "title",
        [
            "Rozdział 1",
            "Rozdział I",
            "ROZDZIAŁ XII",
            "rozdzial 5",                  # bez diakrytyków
            "Chapter 1",
            "Chapter VII",
            "Część 2",
            "Część pierwsza",
            "Part III",
            "Prolog",
            "Prologue",
            "Epilog",
            "Wstęp",
            "Introduction",
        ],
    )
    def test_numerowane_rozdzialy_zawsze_zostaja(self, title: str):
        f = ChapterFilter()
        skip, reason = f.should_skip(title, "<p>treść</p>", "treść" * 50)
        assert skip is False, f"Whitelist powinien zachować '{title}'"
        assert reason is None

    def test_whitelist_wygrywa_z_linkami(self):
        """Nawet jeśli rozdział 1 ma jakieś linki - to wciąż rozdział."""
        html = "<p>Treść z " + "<a href='#'>link</a> " * 20 + "tekstem.</p>"
        text = "Treść " + "link " * 20 + "z dodatkowymi słowami " * 10
        f = ChapterFilter()
        skip, _ = f.should_skip("Rozdział 1", html, text)
        assert skip is False


class TestChapterFilterBlacklist:
    """Tytuły zdradzające sekcje techniczne książki."""

    @pytest.mark.parametrize(
        "title",
        [
            "Spis treści",
            "Spis Treści",
            "SPIS TREŚCI",
            "Spis rozdziałów",
            "Dedykacja",
            "Podziękowania",
            "O autorze",
            "Nota o autorze",
            "Bibliografia",
            "Przypisy",
            "Indeks",
            "Prawa autorskie",
            "Copyright",
            "Table of Contents",
            "Contents",
            "Acknowledgments",
            "Dedication",
            "About the Author",
            "Bibliography",
            "Index",
            "Colophon",
            "Title Page",
        ],
    )
    def test_blacklist_pomija(self, title: str):
        f = ChapterFilter()
        skip, reason = f.should_skip(title, "<p>treść</p>", "treść normalna " * 30)
        assert skip is True, f"Blacklist powinien pominąć '{title}'"
        assert reason is not None and len(reason) > 0


class TestChapterFilterLinkDensity:
    """Heurystyka: TOC w przebraniu - dużo linków, mało treści narracyjnej."""

    def test_wysoka_gestosc_linkow_pomijana(self):
        # 50 anchorów, każdy ~3 słowa = 150 link-words. Total ~200 słów.
        html = (
            "<ol>"
            + "".join(
                f"<li><a href='#c{i}'>Rozdział {i} pierwszy drugi</a></li>"
                for i in range(50)
            )
            + "</ol>"
        )
        text = " ".join(f"Rozdział {i} pierwszy drugi" for i in range(50))
        f = ChapterFilter()
        skip, reason = f.should_skip("Bez tytułu", html, text)
        assert skip is True
        assert "linków" in reason

    def test_niska_gestosc_linkow_zachowana(self):
        # Akapit z 1 linkiem na ~100 słów - to normalny rozdział.
        text = "To jest normalny rozdział z dużą ilością tekstu " * 20
        html = f"<p>{text} Tu jest <a href='#'>jeden link</a> i nic więcej.</p>"
        f = ChapterFilter()
        skip, _ = f.should_skip("Rozdział narracyjny", html, text)
        assert skip is False

    def test_krotki_tekst_pomija_heurystyke_link_density(self):
        """Dla < 30 słów heurystyka link-density jest wyłączona."""
        html = "<p><a href='#'>Link</a> w środku krótkiego.</p>"
        text = "Link w środku krótkiego."  # 4 słowa
        f = ChapterFilter()
        skip, _ = f.should_skip("Coś krótkiego", html, text)
        # 1 link-word / 4 total = 25% - powinno by przeszło limit 30%,
        # ale za mało słów żeby heurystyka się aktywowała.
        assert skip is False

    def test_konfigurowany_prog_link_density(self):
        # 5 linków po 1 słowie z 50 słów = 10% density.
        text = "słowo " * 50
        html = "<p>" + " ".join("<a href='#'>link</a>" for _ in range(5)) + " " + text + "</p>"
        # Z domyślnym progiem 30% - zostaje.
        f_default = ChapterFilter()
        skip, _ = f_default.should_skip("Coś", html, text + " link link link link link")
        assert skip is False
        # Z bardzo niskim progiem - pomijamy.
        f_strict = ChapterFilter(max_link_density=0.05)
        skip, reason = f_strict.should_skip(
            "Coś", html, text + " link link link link link"
        )
        assert skip is True

    def test_niepoprawny_prog_rzuca_value_error(self):
        with pytest.raises(ValueError, match="max_link_density"):
            ChapterFilter(max_link_density=0.0)
        with pytest.raises(ValueError, match="max_link_density"):
            ChapterFilter(max_link_density=1.5)


class TestChapterFilterIntegracja:
    """End-to-end - parser z filterem na realistycznym EPUB."""

    def _make_realistic_epub(self, tmp_path: Path) -> Path:
        """EPUB z mieszanką: TOC, copyright, normalne rozdziały, indeks."""
        return _build_minimal_epub(
            tmp_path,
            [
                # 1. TOC - pełen linków, bez słowa kluczowego w tytule
                "<h1>Bez tytułu</h1><ol>"
                + "".join(
                    f"<li><a href='#c{i}'>Rozdział {i} pierwszy drugi trzeci</a></li>"
                    for i in range(30)
                )
                + "</ol>",
                # 2. Copyright - poznany po tytule
                "<h1>Copyright</h1><p>Wszelkie prawa zastrzeżone. "
                "Wydawnictwo XYZ 2024 oraz dodatkowy tekst dla progu.</p>",
                # 3. Dedykacja
                "<h1>Dedykacja</h1><p>Dla mojej żony za jej cierpliwość "
                "i wsparcie w trakcie pisania tej książki.</p>",
                # 4. Pierwszy prawdziwy rozdział
                "<h1>Rozdział 1</h1><p>Pan Tadeusz wstał wcześnie rano. "
                "Spojrzał przez okno na horyzont. Pomyślał o nadchodzącym dniu.</p>",
                # 5. Drugi rozdział
                "<h1>Rozdział 2</h1><p>Następnego dnia wybrał się na "
                "spacer. Las pachniał jesienią. Liście szeleściły pod butami.</p>",
                # 6. Przypisy
                "<h1>Przypisy</h1><p>1. Patrz literatura przedmiotu. "
                "2. Cytat z Norwida w tłumaczeniu autora niniejszej książki.</p>",
                # 7. O autorze
                "<h1>O autorze</h1><p>Jan Kowalski urodził się w 1970 "
                "roku. Jest absolwentem polonistyki Uniwersytetu Jagiellońskiego.</p>",
            ],
        )

    def test_realistyczny_epub_pomija_smieci_zostawia_rozdzialy(
        self, tmp_path: Path
    ):
        epub_path = self._make_realistic_epub(tmp_path)
        parser = EpubParser(epub_path)

        chapters = list(parser.iter_chapters())

        # Z 7 sekcji powinny zostać tylko 2 prawdziwe rozdziały.
        assert len(chapters) == 2
        titles = [c.title for c in chapters]
        assert "Rozdział 1" in titles
        assert "Rozdział 2" in titles
        # Nie ma śmieci
        assert not any("Copyright" in t for t in titles)
        assert not any("Dedykacja" in t for t in titles)
        assert not any("Przypisy" in t for t in titles)
        assert not any("autorze" in t for t in titles)

    def test_skip_garbage_False_zachowuje_wszystko(self, tmp_path: Path):
        """Backward-compat - wyłączamy heurystykę i mamy stare zachowanie."""
        epub_path = self._make_realistic_epub(tmp_path)
        parser = EpubParser(epub_path)

        chapters = list(parser.iter_chapters(skip_garbage=False))

        # Bez heurystyki wszystkie 7 sekcji przechodzi (są wszystkie > 50 znaków).
        assert len(chapters) == 7

    def test_wlasna_instancja_chapter_filter(self, tmp_path: Path):
        """Można podać własny filter z restrykcyjnym link-density."""
        epub_path = self._make_realistic_epub(tmp_path)
        parser = EpubParser(epub_path)

        # Restrykcyjny filter
        my_filter = ChapterFilter(max_link_density=0.05)
        chapters = list(parser.iter_chapters(chapter_filter=my_filter))

        # Wciąż dwa rozdziały - normalne rozdziały nie mają linków.
        assert len(chapters) == 2

    def test_indeksacja_jest_ciagla_po_skipach(self, tmp_path: Path):
        """Po wycięciu śmieci indeksy 0..N-1 są bez dziur."""
        epub_path = self._make_realistic_epub(tmp_path)
        parser = EpubParser(epub_path)

        chapters = list(parser.iter_chapters())
        indices = [c.index for c in chapters]
        assert indices == list(range(len(chapters)))
