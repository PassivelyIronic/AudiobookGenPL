"""
Text Chunker — dzielenie tekstu rozdziału na fragmenty do TTS.

Strategia:
    1. Najpierw "maskujemy" kropki w polskich skrótach (np., m.in., prof.)
       żeby silnik podziału na zdania ich nie traktował jako koniec zdania.
    2. Dzielimy tekst na zdania regexem: znak terminujący (.!?…) + biały
       znak.
    3. Sklejamy zdania w chunki nie przekraczające `max_chars`.
    4. Jeśli pojedyncze zdanie jest zbyt długie - tniemy je po przecinkach
       i myślnikach. W ostateczności - po słowach (NIGDY w środku słowa).

Wynik: lista (lub generator) chunków idealnie pasujących do limitu kontekstu
modelu TTS (XTTS-v2 sugeruje max ~250-400 znaków na input; ustawiamy
bezpieczny domyślny limit 500).
"""
from __future__ import annotations

import re
from typing import Final, Iterator

# Polskie skróty (i kilka łacińskich), po których kropka NIE oznacza
# końca zdania. Lista nie jest wyczerpująca - można rozszerzać w razie
# błędów w testach na realnych książkach.
POLISH_ABBREVIATIONS: Final[frozenset[str]] = frozenset(
    {
        # ogólne
        "np", "tj", "tzn", "tzw", "itd", "itp", "etc", "cdn", "ww", "ds",
        "m.in", "tj", "wg", "ok", "ca", "vs", "tzw",
        # tytuły, stopnie
        "prof", "dr", "mgr", "inż", "hab", "lic", "płk", "ppłk", "mjr",
        "kpt", "por", "ppor", "gen", "sierż",
        # geografia / adresy
        "ul", "al", "pl", "os", "nr", "str", "tel", "faks",
        "płn", "płd", "wsch", "zach",
        # czas / miary
        "godz", "min", "sek", "ms", "kg", "g", "mg", "km", "m", "cm", "mm",
        # religia / honorific
        "św", "bł", "ks", "im",
        # daty
        "r", "w",  # "r." = roku, "w." = wiek
        # różne
        "rys", "tab", "fot", "ryc", "art", "ust", "pkt", "rozdz",
        "ws", "ze",
    }
)


class TextChunker:
    """
    Tnie tekst na fragmenty <= max_chars, szanując granice zdań i słów.

    Klasa jest bezstanowa (poza konfiguracją z konstruktora), więc jedną
    instancję można reużywać dla wielu rozdziałów.
    """

    # Granica zdania: znak terminujący + co najmniej jeden biały znak.
    # Lookbehind zachowuje znak terminujący na końcu poprzedniego zdania.
    _SENTENCE_SPLIT_RE: Final[re.Pattern[str]] = re.compile(
        r"(?<=[.!?…])\s+"
    )
    # Granica klauzuli wewnątrz zdania - cięcie na potrzeby długich zdań.
    _CLAUSE_SPLIT_RE: Final[re.Pattern[str]] = re.compile(
        r"(?<=[,;:—–])\s+"
    )
    # Znacznik używany do tymczasowego "maskowania" kropek w skrótach.
    _MASK: Final[str] = "\x00"

    def __init__(self, max_chars: int = 500) -> None:
        if max_chars < 20:
            raise ValueError(
                f"max_chars musi wynosić co najmniej 20, otrzymano {max_chars}"
            )
        self._max_chars: int = max_chars
        # Pre-kompilacja regexów maskujących dla każdego skrótu - robione raz.
        self._abbr_patterns: list[re.Pattern[str]] = [
            re.compile(rf"\b{re.escape(abbr)}\.", re.IGNORECASE)
            for abbr in POLISH_ABBREVIATIONS
        ]

    # ----- API publiczne -----------------------------------------------------

    @property
    def max_chars(self) -> int:
        return self._max_chars

    def chunk(self, text: str) -> list[str]:
        """Materializuje wszystkie chunki do listy."""
        return list(self.iter_chunks(text))

    def iter_chunks(self, text: str) -> Iterator[str]:
        """Generator chunków (oszczędny pamięciowo dla długich rozdziałów)."""
        text = (text or "").strip()
        if not text:
            return

        sentences = self._split_into_sentences(text)

        buffer: list[str] = []
        buffer_len = 0

        for sentence in sentences:
            sent_len = len(sentence)

            # Pojedyncze zdanie nie mieści się w limicie — łamiemy je osobno.
            if sent_len > self._max_chars:
                if buffer:
                    yield " ".join(buffer)
                    buffer, buffer_len = [], 0
                yield from self._split_long_sentence(sentence)
                continue

            # +1 na spację, jeśli bufor już coś zawiera.
            projected = buffer_len + sent_len + (1 if buffer else 0)
            if projected > self._max_chars:
                yield " ".join(buffer)
                buffer, buffer_len = [sentence], sent_len
            else:
                buffer.append(sentence)
                buffer_len = projected

        if buffer:
            yield " ".join(buffer)

    # ----- Wewnętrzne --------------------------------------------------------

    def _split_into_sentences(self, text: str) -> list[str]:
        """Dzieli tekst na zdania, omijając kropki w skrótach."""
        masked = self._mask_abbreviations(text)
        parts = self._SENTENCE_SPLIT_RE.split(masked)
        return [
            self._unmask(part).strip()
            for part in parts
            if part and part.strip()
        ]

    def _mask_abbreviations(self, text: str) -> str:
        for pattern in self._abbr_patterns:
            text = pattern.sub(lambda m: m.group(0)[:-1] + self._MASK, text)
        return text

    @classmethod
    def _unmask(cls, text: str) -> str:
        return text.replace(cls._MASK, ".")

    def _split_long_sentence(self, sentence: str) -> Iterator[str]:
        """
        Tnie zbyt długie zdanie po przecinkach/dwukropkach/myślnikach.
        Jeśli nadal któraś klauzula nie mieści się - przechodzi na cięcie
        po słowach.
        """
        clauses = self._CLAUSE_SPLIT_RE.split(sentence)

        buffer: list[str] = []
        buffer_len = 0

        for clause in clauses:
            clause = clause.strip()
            if not clause:
                continue

            clen = len(clause)
            if clen > self._max_chars:
                if buffer:
                    yield " ".join(buffer)
                    buffer, buffer_len = [], 0
                yield from self._split_by_words(clause)
                continue

            projected = buffer_len + clen + (1 if buffer else 0)
            if projected > self._max_chars:
                yield " ".join(buffer)
                buffer, buffer_len = [clause], clen
            else:
                buffer.append(clause)
                buffer_len = projected

        if buffer:
            yield " ".join(buffer)

    def _split_by_words(self, text: str) -> Iterator[str]:
        """
        Ostatnia deska ratunku: tnie po granicach słów.

        Jeśli pojedyncze słowo i tak przekracza limit (rzadkie - np. URL),
        yieldujemy je w całości BEZ DZIELENIA - lepiej oddać TTS odrobinę
        za długi chunk niż uciąć słowo w połowie.
        """
        words = text.split()
        buffer: list[str] = []
        buffer_len = 0

        for word in words:
            wlen = len(word)
            projected = buffer_len + wlen + (1 if buffer else 0)

            if projected > self._max_chars and buffer:
                yield " ".join(buffer)
                buffer, buffer_len = [word], wlen
            else:
                buffer.append(word)
                buffer_len = projected

        if buffer:
            yield " ".join(buffer)
