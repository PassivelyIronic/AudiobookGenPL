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
from typing import Iterator

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

    def iter_chapters(self, min_chars: int = 50) -> Iterator[Chapter]:
        """
        Generator zwracający kolejne rozdziały książki.

        Args:
            min_chars: minimalna długość tekstu rozdziału, by trafił do
                wyjścia. Krótsze fragmenty (strony tytułowe, copyright,
                spis treści) są pomijane.

        Yields:
            Chapter: rozdział z indeksem, tytułem i czystym tekstem.

        Raises:
            EpubParserError: gdy biblioteka nie zdoła otworzyć pliku.
        """
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
