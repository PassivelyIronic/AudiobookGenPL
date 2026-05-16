"""
EPUB Parser — strumieniowe wyciąganie rozdziałów z pliku .epub.

Ze względu na limit pamięci (~4.8 GB RAM) parser udostępnia interfejs
generatora: rozdziały są yieldowane pojedynczo, dzięki czemu pipeline
TTS może przetworzyć i zapisać audio rozdziału na dysk, zanim załaduje
kolejny.

Założenia:
    * Pliki EPUB pochodzą z konwersji w Calibre, więc są poprawne
      strukturalnie (brak heroicznej obsługi błędów formatu).
    * Wyjściem jest CZYSTY tekst (bez HTML), znormalizowany pod kątem
      białych znaków, gotowy do podziału na chunki TTS.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Iterator

from bs4 import BeautifulSoup
from ebooklib import ITEM_DOCUMENT, epub

logger = logging.getLogger(__name__)


# --- Wyjątki -----------------------------------------------------------------


class EpubParserError(Exception):
    """Błąd parsowania pliku EPUB (brak pliku, zła struktura itp.)."""


# --- Model danych ------------------------------------------------------------


@dataclass(frozen=True)
class Chapter:
    """Pojedynczy rozdział wyciągnięty z EPUB-a."""

    index: int
    title: str
    text: str

    def __len__(self) -> int:
        return len(self.text)


# --- Heurystyka filtrowania śmieciowych rozdziałów ---------------------------


class ChapterFilter:
    """
    Decyduje, czy rozdział jest "śmieciowy" (TOC, copyright, dedykacja, etc.)
    i powinien zostać pominięty w syntezie audio.

    Strategia od najwęższego do najszerszego sygnału:
        1. WHITELIST (zawsze zachowaj) - tytuły matchujące wzorzec rozdziału
           numerowanego ("Rozdział 5", "Chapter VII", "Część II"). Numer to
           bardzo mocny sygnał, że to TREŚĆ.
        2. BLACKLIST po słowie kluczowym w tytule - znane wzorce stron
           technicznych książki.
        3. LINK-DENSITY heurystyka - rozdziały, w których > 30% słów to
           anchor text, to praktycznie zawsze TOC (każda pozycja = link).

    Implementacja jest świadomie zachowawcza: w razie wątpliwości
    ZACHOWUJEMY rozdział - lepiej zsyntezować copyright (10 sekund audio)
    niż wyciąć prawdziwy rozdział.
    """

    # Wzorzec: "Rozdział 5", "Chapter VII", "Część 2", "Prolog" (case-insensitive).
    # Lookbehind dla \b nie działa w Pythonie, więc strażnik na początku
    # poprzez `^\s*`. Akceptujemy cyfry arabskie I rzymskie.
    _ALWAYS_KEEP_RE: Final[re.Pattern[str]] = re.compile(
        r"^\s*("
        r"rozdział|rozdzial|chapter|część|czesc|part|book|tom|"
        r"prolog|prologue|epilog|epilogue|wstęp|wstep|introduction"
        r")\s*"
        r"([\divxlcmIVXLCM]+|pierwszy|drugi|trzeci|czwarty|piąty|"
        r"first|second|third|fourth|fifth)?",
        re.IGNORECASE | re.UNICODE,
    )

    # Słowa kluczowe w tytule, których obecność = pomijamy (chyba że
    # ALWAYS_KEEP też matchnęło - whitelist wygrywa).
    _BLACKLIST_KEYWORDS: Final[frozenset[str]] = frozenset({
        # Polski
        "spis treści", "spis tresci", "spis rozdziałów", "spis rozdzialow",
        "dedykacja", "podziękowania", "podziekowania", "o autorze",
        "o autorce", "nota o autorze", "przypisy", "indeks", "skorowidz",
        "bibliografia", "literatura", "strona tytułowa", "strona tytulowa",
        "metryczka", "prawa autorskie", "copyright", "kolofon", "stopka",
        # Angielski (część książek polskich ma sekcje angielskie)
        "table of contents", "contents", "toc", "acknowledgments",
        "acknowledgements", "dedication", "about the author", "about author",
        "notes", "endnotes", "footnotes", "bibliography", "references",
        "index", "colophon", "imprint", "title page", "halftitle",
        "front matter", "back matter",
    })

    DEFAULT_MAX_LINK_DENSITY: Final[float] = 0.30
    # Minimum słów żeby w ogóle stosować heurystykę link-density (dla 5-słowowego
    # rozdziału jeden link = 20%, ale to bez sensu liczyć).
    _LINK_DENSITY_MIN_WORDS: Final[int] = 30

    def __init__(self, max_link_density: float = DEFAULT_MAX_LINK_DENSITY) -> None:
        if not 0.0 < max_link_density <= 1.0:
            raise ValueError(
                f"max_link_density musi być w (0, 1], otrzymano {max_link_density}"
            )
        self._max_link_density = max_link_density

    def should_skip(
        self, title: str, html: str, text: str
    ) -> tuple[bool, str | None]:
        """
        Czy ten rozdział pominąć?

        Returns:
            (skip, reason) - jeśli `skip=True`, `reason` jest niepustym
            stringiem nadającym się do zalogowania.
        """
        title_norm = title.lower().strip()

        # 1. WHITELIST - rozdział numerowany ZAWSZE zostaje.
        if self._ALWAYS_KEEP_RE.match(title_norm):
            return False, None

        # 2. BLACKLIST - słowa kluczowe w tytule.
        for keyword in self._BLACKLIST_KEYWORDS:
            if keyword in title_norm:
                return True, f"tytuł sugeruje sekcję techniczną ('{keyword}')"

        # 3. LINK-DENSITY - heurystyka na podstawie struktury HTML.
        density = self._compute_link_density(html, text)
        if density > self._max_link_density:
            return True, f"za wysoka gęstość linków ({density:.0%})"

        return False, None

    def _compute_link_density(self, html: str, text: str) -> float:
        """
        Oblicza, jaki procent słów w rozdziale to anchor text.

        Działa na surowym HTML (przed _clean_html), żeby tagi <a> były
        jeszcze obecne. Zwraca 0.0 dla zbyt krótkich tekstów (poniżej
        progu sensowności).
        """
        total_words = len(text.split())
        if total_words < self._LINK_DENSITY_MIN_WORDS:
            return 0.0

        try:
            soup = BeautifulSoup(html, "lxml")
        except Exception:  # noqa: BLE001 - lxml rzadko, ale potrafi
            return 0.0

        link_words = 0
        for anchor in soup.find_all("a"):
            link_text = anchor.get_text(separator=" ", strip=True)
            link_words += len(link_text.split())

        return link_words / total_words


# --- Parser ------------------------------------------------------------------


class EpubParser:
    """
    Czyta plik EPUB i yielduje rozdziały po jednym na raz.

    Użycie:
        parser = EpubParser("ksiazka.epub")
        for chapter in parser.iter_chapters():
            ...  # przetwarzaj rozdział, potem pamięć jest zwalniana
    """

    # Tagi, których zawartość nigdy nie powinna trafić do audiobooka.
    _STRIPPABLE_TAGS: tuple[str, ...] = (
        "script",
        "style",
        "nav",
        "header",
        "footer",
        "aside",
        "sup",
        "sub",
        "figure",
        "figcaption",
        "img",
    )

    # Wyrażenie regularne łączące wielokrotne białe znaki w pojedynczą spację.
    _WHITESPACE_RE = re.compile(r"\s+")
    # Usuwa spację stojącą tuż przed znakiem interpunkcyjnym.
    _SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([.,;:!?…])")

    def __init__(self, epub_path: str | Path) -> None:
        path = Path(epub_path)
        if not path.is_file():
            raise EpubParserError(f"Plik EPUB nie istnieje: {path}")
        if path.suffix.lower() != ".epub":
            raise EpubParserError(f"Plik nie ma rozszerzenia .epub: {path}")
        self._path: Path = path

    # ----- API publiczne -----------------------------------------------------

    def iter_chapters(
        self,
        min_chars: int = 50,
        skip_garbage: bool = True,
        chapter_filter: ChapterFilter | None = None,
    ) -> Iterator[Chapter]:
        """
        Generator zwracający kolejne rozdziały książki.

        Args:
            min_chars: minimalna długość tekstu rozdziału, by trafił do
                wyjścia. Krótsze fragmenty (strony tytułowe, copyright,
                spis treści) są pomijane.
            skip_garbage: czy pomijać śmieciowe rozdziały (TOC, dedykacje,
                copyright itp.) na podstawie heurystyki ChapterFilter.
                Domyślnie True - oszczędza godziny syntezy TTS.
            chapter_filter: opcjonalna własna instancja ChapterFilter
                (np. z innym progiem link-density). Domyślnie tworzymy
                z defaultami.

        Yields:
            Chapter: rozdział z indeksem, tytułem i czystym tekstem.

        Raises:
            EpubParserError: gdy biblioteka nie zdoła otworzyć pliku.
        """
        if skip_garbage and chapter_filter is None:
            chapter_filter = ChapterFilter()
        elif not skip_garbage:
            chapter_filter = None

        try:
            book = epub.read_epub(str(self._path))
        except Exception as exc:  # ebooklib rzuca różnymi wyjątkami
            raise EpubParserError(
                f"Nie udało się otworzyć pliku EPUB ({self._path}): {exc}"
            ) from exc

        idx = 0
        for item in book.get_items_of_type(ITEM_DOCUMENT):
            try:
                content = item.get_content()
                # EbookLib bywa kapryśny: czasem zwraca bytes (typowo), a
                # czasem już zdekodowany str (np. dla itemów dodanych w
                # pamięci). Obsługujemy oba przypadki.
                if isinstance(content, bytes):
                    raw_html = content.decode("utf-8", errors="replace")
                else:
                    raw_html = str(content)
            except Exception as exc:  # noqa: BLE001 - logujemy i lecimy dalej
                logger.warning(
                    "Pomijam uszkodzony item %s: %s", item.get_name(), exc
                )
                continue

            text = self._clean_html(raw_html)
            if len(text) < min_chars:
                logger.debug(
                    "Pomijam krótki rozdział %s (%d znaków)",
                    item.get_name(),
                    len(text),
                )
                continue

            title = self._extract_title(raw_html) or f"Rozdział {idx + 1}"

            # Heurystyka anty-śmieciowa
            if chapter_filter is not None:
                should_skip, reason = chapter_filter.should_skip(
                    title, raw_html, text
                )
                if should_skip:
                    logger.info(
                        "Pomijam śmieciowy rozdział '%s' (%s)", title, reason
                    )
                    continue

            yield Chapter(index=idx, title=title, text=text)
            idx += 1

    # ----- Pomocnicze (chronione - testowane przez iter_chapters) ------------

    def _clean_html(self, html: str) -> str:
        """Wycina niechciane tagi i zwraca znormalizowany tekst."""
        soup = BeautifulSoup(html, "lxml")

        for tag_name in self._STRIPPABLE_TAGS:
            for tag in soup.find_all(tag_name):
                tag.decompose()

        # separator=" " gwarantuje, że tagi blokowe (np. <p>, <h1>) nie
        # skleją się w jeden ciąg bez spacji.
        raw_text = soup.get_text(separator=" ")
        return self._normalize_whitespace(raw_text)

    @classmethod
    def _normalize_whitespace(cls, text: str) -> str:
        text = cls._WHITESPACE_RE.sub(" ", text)
        text = cls._SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)
        return text.strip()

    @staticmethod
    def _extract_title(html: str) -> str | None:
        """Próbuje znaleźć tytuł rozdziału w nagłówku h1/h2/h3."""
        soup = BeautifulSoup(html, "lxml")
        for level in ("h1", "h2", "h3"):
            tag = soup.find(level)
            if tag and tag.get_text(strip=True):
                return tag.get_text(strip=True)
        return None
