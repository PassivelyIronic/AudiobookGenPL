"""
Testy jednostkowe dla TextChunker.

Pokrywają:
    * limity max_chars (żaden chunk ich nie przekracza),
    * brak ucinania słów (każde słowo z wejścia obecne w wyjściu w całości),
    * cięcie na granicach zdań (kropka, ?, !, …),
    * polskie skróty (np., m.in., prof., dr) - nie powodują rozcięcia,
    * długie zdania bez końca terminującego (cięcie po przecinkach),
    * konfigurację (walidacja max_chars),
    * idempotentność listy vs. iteratora.
"""
from __future__ import annotations

import re

import pytest

from app.services.text_chunker import POLISH_ABBREVIATIONS, TextChunker


# ============================================================
#  Podstawy
# ============================================================


class TestPodstawowe:
    def test_pusty_tekst_zwraca_pusta_liste(self):
        assert TextChunker().chunk("") == []

    def test_same_biale_znaki_zwracaja_pusta_liste(self):
        assert TextChunker().chunk("   \n\t  ") == []

    def test_krotki_tekst_to_jeden_chunk(self):
        chunker = TextChunker(max_chars=500)
        text = "To jest krótkie zdanie. I drugie."
        assert chunker.chunk(text) == [text]

    def test_zachowanie_polskich_znakow(self):
        chunker = TextChunker(max_chars=500)
        text = "Zażółć gęślą jaźń. Łódź to miasto. Pójdę nad Bałtyk."
        assert chunker.chunk(text) == [text]


# ============================================================
#  Limity rozmiaru
# ============================================================


class TestLimity:
    @pytest.mark.parametrize("max_chars", [80, 150, 300, 500, 1000])
    def test_zaden_chunk_nie_przekracza_limitu(self, max_chars: int):
        chunker = TextChunker(max_chars=max_chars)
        text = (
            "Pierwsze zdanie z polskimi znakami ąęć. "
            "Drugie zdanie - również z polskimi żźń. "
            "Trzecie zdanie kończące akapit ółĄ. "
            "Czwarte dla porządku - takie sobie. "
            "Piąte na koniec testu, dla pewności. "
        ) * 5
        for chunk in chunker.chunk(text):
            assert len(chunk) <= max_chars, (
                f"Chunk za długi ({len(chunk)} > {max_chars}): {chunk!r}"
            )

    def test_max_chars_ponizej_minimum_rzuca_wyjatek(self):
        with pytest.raises(ValueError, match="max_chars"):
            TextChunker(max_chars=5)


# ============================================================
#  Granice słów - CHUNKER NIE MOŻE UCINAĆ SŁÓW
# ============================================================


class TestGraniceSlow:
    def test_zadne_slowo_nie_jest_uciete(self):
        chunker = TextChunker(max_chars=60)
        text = (
            "Konstantynopolitańczykowianeczka rozmawiała z przyjaciółmi. "
            "Niezidentyfikowany obiekt latający przeleciał nad miastem. "
            "Charakterystyczna intonacja zwróciła uwagę słuchaczy."
        )
        chunks = chunker.chunk(text)
        original_words = text.split()
        joined = " ".join(chunks)
        chunked_words = joined.split()
        # Każde oryginalne słowo musi się znaleźć w wyjściu bez okaleczeń.
        for word in original_words:
            assert word in chunked_words, (
                f"Słowo '{word}' zostało okaleczone lub zgubione"
            )

    def test_chunki_zaczynaja_i_koncza_pelnymi_slowami(self):
        chunker = TextChunker(max_chars=70)
        text = (
            "Pierwsze zdanie ma kilka słów. "
            "Drugie zdanie również jest pełne treści. "
            "Trzecie zdanie kończy akapit pewnie i dobitnie."
        )
        for chunk in chunker.chunk(text):
            # żaden chunk nie zaczyna i nie kończy się w środku słowa
            assert not chunk.startswith(" ")
            assert not chunk.endswith(" ")
            # litera na początku/końcu nie wisi obok litery z poprzedniego
            # słowa - prosty sanity check: chunk to ciąg słów rozdzielonych
            # pojedynczymi spacjami
            assert "  " not in chunk

    def test_brak_zgubionych_znakow(self):
        """Suma długości słów we wszystkich chunkach == suma w oryginale."""
        chunker = TextChunker(max_chars=90)
        text = (
            "Krótkie zdanie. Średnie zdanie z większą liczbą słów. "
            "Naprawdę bardzo długie zdanie, z licznymi przecinkami, "
            "podrzędnymi członami i wszelkimi dodatkami, które razem "
            "tworzą rozbudowaną wypowiedź godną prozy XIX wieku."
        )
        chunks = chunker.chunk(text)
        orig_chars = sum(len(w) for w in text.split())
        new_chars = sum(len(w) for c in chunks for w in c.split())
        assert orig_chars == new_chars


# ============================================================
#  Granice zdań
# ============================================================


class TestGraniceZdan:
    def test_chunki_koncza_sie_terminatorem_zdania(self):
        chunker = TextChunker(max_chars=80)
        text = (
            "Pierwsze zdanie. "
            "Drugie zdanie kończy się tu. "
            "Trzecie pyta o coś ciekawego? "
            "Czwarte krzyczy głośno! "
            "Piąte zwyczajne na koniec."
        )
        chunks = chunker.chunk(text)
        terminators = (".", "!", "?", "…")
        # Wszystkie chunki oprócz ewentualnego ostatniego (gdyby tekst
        # nie kończył się terminatorem) muszą kończyć terminatorem.
        for chunk in chunks:
            assert chunk[-1] in terminators, (
                f"Chunk nie kończy się terminatorem zdania: {chunk!r}"
            )

    def test_wykrzyknik_i_pytajnik_dziala_jak_kropka(self):
        # max_chars=20 - każde zdanie ma ~11 znaków, więc dwa nie zmieszczą
        # się razem z separatorem (11+1+11=23 > 20). Wymusza 3 chunki.
        chunker = TextChunker(max_chars=20)
        text = "Czy działa? Tak działa! Świetnie."
        chunks = chunker.chunk(text)
        assert len(chunks) == 3
        assert chunks[0].endswith("?")
        assert chunks[1].endswith("!")
        assert chunks[2].endswith(".")

    def test_wielokropek_jako_granica_zdania(self):
        # max_chars=25 wymusza rozdział; całość ma 38 znaków.
        chunker = TextChunker(max_chars=25)
        text = "Patrzył w niebo… I myślał o tym długo."
        chunks = chunker.chunk(text)
        assert len(chunks) == 2
        assert chunks[0].endswith("…")


# ============================================================
#  Polskie skróty - NIE wolno na nich ciąć
# ============================================================


class TestPolskieSkroty:
    @pytest.mark.parametrize(
        "text",
        [
            "Lubię owoce, np. jabłka, gruszki i wiśnie.",
            "Zajmuje się m.in. analityką danych w firmie.",
            "Wykład prowadził prof. dr hab. Jan Kowalski wczoraj.",
            "Mieszkam na ul. Kwiatowej 5 w Warszawie obecnie.",
            "Spotkanie o godz. 14:30 w sali nr 12 zaczyna się dziś.",
            "Wydarzenia z 1989 r. zmieniły kraj na zawsze.",
            "Książkę napisał ks. prof. Tischner w latach dziewięćdziesiątych.",
        ],
    )
    def test_zdania_ze_skrotami_nie_sa_rozcinane(self, text: str):
        chunker = TextChunker(max_chars=500)
        chunks = chunker.chunk(text)
        assert len(chunks) == 1, (
            f"Chunker niepotrzebnie pociął zdanie ze skrótem: {chunks}"
        )
        assert chunks[0] == text

    def test_skrot_nie_wciaga_kolejnego_zdania(self):
        """Po skrócie + kolejnym ZDANIU normalna granica musi działać."""
        chunker = TextChunker(max_chars=80)
        text = (
            "Spotkanie odbyło się np. w marcu. "
            "Później dyskutowali jeszcze długo o szczegółach."
        )
        chunks = chunker.chunk(text)
        assert len(chunks) == 2

    def test_lista_skrotow_zawiera_najczestsze(self):
        # Sanity check, gdyby ktoś przypadkiem usunął te zwroty.
        for required in ("np", "m.in", "prof", "dr", "tj", "itd"):
            assert required in POLISH_ABBREVIATIONS


# ============================================================
#  Zdania dłuższe od limitu
# ============================================================


class TestDlugieZdania:
    def test_zdanie_dluzsze_od_limitu_dzielone_po_przecinkach(self):
        chunker = TextChunker(max_chars=60)
        text = (
            "Pierwsze, drugie, trzecie, czwarte, piąte, szóste, "
            "siódme, ósme, dziewiąte oraz dziesiąte słowo kluczowe."
        )
        chunks = chunker.chunk(text)
        assert len(chunks) > 1
        for c in chunks:
            assert len(c) <= 60

    def test_ekstremalnie_dlugie_slowo_nie_psuje_chunkera(self):
        """Słowo dłuższe od limitu jest yieldowane w całości - bez ucinania."""
        chunker = TextChunker(max_chars=50)
        long_word = "A" * 200
        text = f"Słowo {long_word} następne. Koniec."
        chunks = chunker.chunk(text)
        assert len(chunks) >= 1
        # Słowo musi być nietknięte w wyjściu.
        assert long_word in " ".join(chunks)


# ============================================================
#  Spójność interfejsów
# ============================================================


class TestInterfejs:
    def test_lista_i_iterator_zwracaja_to_samo(self):
        chunker = TextChunker(max_chars=120)
        text = "Pierwsze zdanie. Drugie. Trzecie zdanie testowe. " * 8
        assert chunker.chunk(text) == list(chunker.iter_chunks(text))

    def test_iter_chunks_to_generator(self):
        import types

        chunker = TextChunker()
        result = chunker.iter_chunks("Zdanie testowe.")
        assert isinstance(result, types.GeneratorType)


# ============================================================
#  Realistyczny mini-rozdział
# ============================================================


class TestRealistyczny:
    def test_pelny_akapit_w_jezyku_polskim(self):
        chunker = TextChunker(max_chars=200)
        text = (
            "Pan Tadeusz wstał o godz. 7 rano. Spojrzał przez okno - na "
            "horyzoncie majaczyło Wilno. Pomyślał o przyjaciołach, np. o "
            "Jacku Soplicy, którego nie widział od lat. \"Czas wracać\" - "
            "powiedział sam do siebie. Wziął kapelusz i wyszedł. Droga "
            "była długa, ale pełna nadziei na nowe spotkania!"
        )
        chunks = chunker.chunk(text)
        # Spójność: wszystkie słowa zachowane
        orig_words = text.split()
        joined_words = " ".join(chunks).split()
        assert orig_words == joined_words
        # Każdy chunk pod limitem
        for c in chunks:
            assert len(c) <= 200
        # "np." nie spowodowało rozcięcia w połowie zdania ze Soplicą
        for c in chunks:
            if "np." in c:
                # po "np." powinien być kontekst, a nie koniec chunku
                assert not re.search(r"np\.\s*$", c)
