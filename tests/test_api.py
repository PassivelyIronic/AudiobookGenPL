"""
Testy API FastAPI.

Strategia testowania bez Redisa:
    Celery w trybie `task_always_eager=True` wykonuje taski synchronicznie
    w procesie testowym. Nie potrzeba brokera ani backendu - cały pipeline
    przepływa w czasie wywołania endpointu /upload.

Limitacje eager mode:
    * `task.delay()` zwraca natychmiast wynik, więc /status testujemy
      tylko dla stanu SUCCESS i FAILURE - PROGRESS w eager mode nie
      występuje (nie ma osobnego workera, który by aktualizował state).
    * Sygnał worker_process_init też się nie wywoła - jeśli task tego
      potrzebuje, musi zrobić lazy load.

Pokrywają:
    * happy path: upload .epub → task SUCCESS → /status zwraca ścieżkę MP3,
    * walidacja: brak pliku, złe rozszerzenie, pusty plik, za duży plik,
    * status nieznanego taska → PENDING,
    * health + root.
"""
from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import patch

import pytest
from ebooklib import epub
from fastapi.testclient import TestClient

from app.core.config import Settings, get_settings
from app.core.tts_model import SingletonMeta, TTSModel
from app.main import app
from app.worker import celery_app


# ============================================================
#  Fixtures - settings nadpisany na tmp_path, Celery eager
# ============================================================


@pytest.fixture(autouse=True)
def _reset_state():
    SingletonMeta._reset(TTSModel)
    get_settings.cache_clear()
    yield
    SingletonMeta._reset(TTSModel)
    get_settings.cache_clear()


@pytest.fixture
def test_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    get_settings.cache_clear()
    
    # 1. Wymuszamy ścieżki w zmiennych środowiskowych, żeby wszystkie
    # moduły (w tym worker Celery) pobierały katalogi testowe.
    monkeypatch.setenv("EN_UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("EN_OUTPUT_DIR", str(tmp_path / "outputs"))
    monkeypatch.setenv("EN_WORK_DIR", str(tmp_path / "work"))
    monkeypatch.setenv("EN_TTS_MOCK_MODE", "true")
    
    # 2. Teraz get_settings() zbuduje i zakeczuje obiekt z naszymi ścieżkami
    s = get_settings()
    
    # 3. Nadpisujemy dla FastAPI (żeby endpoints też z tego korzystały)
    app.dependency_overrides[get_settings] = lambda: s
    return s


@pytest.fixture
def client(test_settings: Settings, monkeypatch):
    """TestClient z nadpisanym settings i Celery w eager mode."""
    # KRYTYCZNE: worker Celery (nawet w eager mode) używa własnego
    # `get_settings()` zaimportowanego do worker.py - dependency_overrides
    # FastAPI go nie widzi. Musimy więc ustawić env vars, żeby worker
    # odczytał te same ścieżki co API. To dokładnie tak, jak zachowa się
    # w produkcji - oba procesy czytają z .env / env vars.
    monkeypatch.setenv("AK_UPLOAD_DIR", str(test_settings.upload_dir))
    monkeypatch.setenv("AK_OUTPUT_DIR", str(test_settings.output_dir))
    monkeypatch.setenv("AK_WORK_DIR", str(test_settings.work_dir))
    monkeypatch.setenv("AK_MAX_UPLOAD_SIZE_MB", str(test_settings.max_upload_size_mb))
    monkeypatch.setenv("AK_TTS_MOCK_MODE", "true")
    monkeypatch.setenv("AK_CHUNK_MAX_CHARS", str(test_settings.chunk_max_chars))
    monkeypatch.setenv("AK_MP3_BITRATE", test_settings.mp3_bitrate)
    get_settings.cache_clear()

    # Eager mode: task.delay() wykonuje się synchronicznie.
    # cache+memory:// zastępuje Redisa - wynik trafia do RAM-u procesu testu,
    # więc AsyncResult w /status/{task_id} go znajdzie.
    original = {
        "task_always_eager": celery_app.conf.task_always_eager,
        "task_eager_propagates": celery_app.conf.task_eager_propagates,
        "task_store_eager_result": celery_app.conf.task_store_eager_result,
        "result_backend": celery_app.conf.result_backend,
    }
    celery_app.conf.update(
        task_always_eager=True,
        task_eager_propagates=False,        # FAILURE state zamiast re-raise
        task_store_eager_result=True,       # zapisz wynik do backendu
        result_backend="cache+memory://",   # in-memory, bez Redisa
    )

    app.dependency_overrides[get_settings] = lambda: test_settings
    try:
        with TestClient(app) as c:
            yield c
    finally:
        app.dependency_overrides.clear()
        celery_app.conf.update(**original)


def _make_epub_bytes(chapters: list[tuple[str, str]]) -> bytes:
    """Buduje EPUB w pamięci i zwraca jego bajty - bez plików tymczasowych."""
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".epub", delete=False) as f:
        tmp_path = Path(f.name)

    try:
        book = epub.EpubBook()
        book.set_identifier("api-test")
        book.set_title("API Test Book")
        book.set_language("pl")
        items = []
        for i, (title, body) in enumerate(chapters, start=1):
            ch = epub.EpubHtml(title=title, file_name=f"c{i}.xhtml", lang="pl")
            ch.set_content(
                f"<html><body><h1>{title}</h1>{body}</body></html>".encode("utf-8")
            )
            book.add_item(ch)
            items.append(ch)
        book.toc = tuple(items)
        book.add_item(epub.EpubNcx())
        book.spine = list(items)
        epub.write_epub(str(tmp_path), book, {})
        return tmp_path.read_bytes()
    finally:
        tmp_path.unlink(missing_ok=True)


@pytest.fixture
def epub_payload() -> bytes:
    return _make_epub_bytes(
        [
            (
                "Rozdział 1",
                "<p>Polski tekst testowy z odpowiednią długością. "
                "Drugie zdanie dla pewności progu min_chars.</p>",
            ),
            (
                "Rozdział 2",
                "<p>Kolejny rozdział, również wystarczająco długi. "
                "Audiobook musi mieć przynajmniej kilka zdań.</p>",
            ),
        ]
    )


# ============================================================
#  Meta endpointy
# ============================================================


class TestMeta:
    def test_root_zwraca_banner(self, client: TestClient):
        r = client.get("/")
        assert r.status_code == 200
        body = r.json()
        assert body["service"] == "audio-ksiaznica"
        assert body["docs"] == "/docs"

    def test_health_zwraca_ok(self, client: TestClient):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


# ============================================================
#  Upload - walidacja
# ============================================================


class TestUploadWalidacja:
    def test_brak_pliku_zwraca_422(self, client: TestClient):
        r = client.post("/upload")
        # FastAPI samo zwraca 422 dla brakującego required field
        assert r.status_code == 422

    def test_zle_rozszerzenie_zwraca_400(self, client: TestClient):
        r = client.post(
            "/upload",
            files={"file": ("plik.txt", b"abc", "text/plain")},
        )
        assert r.status_code == 400
        assert "rozszerzenie" in r.json()["detail"].lower()

    def test_pusty_plik_zwraca_400(self, client: TestClient):
        r = client.post(
            "/upload",
            files={"file": ("pusta.epub", b"", "application/epub+zip")},
        )
        assert r.status_code == 400
        assert "pusty" in r.json()["detail"].lower()

    def test_za_duzy_plik_zwraca_413(self, client: TestClient, test_settings: Settings):
        # max_upload_size_mb = 10, daję 11 MB.
        big_payload = b"x" * (11 * 1024 * 1024)
        r = client.post(
            "/upload",
            files={"file": ("wielki.epub", big_payload, "application/epub+zip")},
        )
        assert r.status_code == 413
        assert "10 MB" in r.json()["detail"]

    def test_path_traversal_w_nazwie_jest_neutralizowany(
        self, client: TestClient, test_settings: Settings, epub_payload: bytes
    ):
        """Nazwa pliku '../../etc/passwd.epub' nie może wyjść poza upload_dir."""
        r = client.post(
            "/upload",
            files={
                "file": (
                    "../../etc/passwd.epub",
                    epub_payload,
                    "application/epub+zip",
                )
            },
        )
        assert r.status_code == 202
        # Plik został zapisany w upload_dir, nie w /etc
        files_in_upload = list(test_settings.upload_dir.iterdir())
        for f in files_in_upload:
            # bezwzględna ścieżka musi być potomkiem upload_dir
            assert test_settings.upload_dir.resolve() in f.resolve().parents \
                or f.parent.resolve() == test_settings.upload_dir.resolve()


# ============================================================
#  Upload - sukces
# ============================================================


class TestUploadSukces:
    def test_poprawny_upload_zwraca_202_i_task_id(
        self, client: TestClient, epub_payload: bytes
    ):
        r = client.post(
            "/upload",
            files={"file": ("ksiazka.epub", epub_payload, "application/epub+zip")},
        )
        assert r.status_code == 202
        body = r.json()
        assert "task_id" in body
        assert body["status_url"] == f"/status/{body['task_id']}"
        assert isinstance(body["task_id"], str)
        assert len(body["task_id"]) > 0


# ============================================================
#  Status - happy path z eager Celery
# ============================================================


class TestStatusEager:
    def test_po_uploadzie_status_jest_SUCCESS(
        self, client: TestClient, epub_payload: bytes
    ):
        # Eager mode: task wykonuje się w czasie POST /upload.
        upload = client.post(
            "/upload",
            files={"file": ("k.epub", epub_payload, "application/epub+zip")},
        )
        task_id = upload.json()["task_id"]

        status = client.get(f"/status/{task_id}")
        assert status.status_code == 200
        body = status.json()
        assert body["task_id"] == task_id
        assert body["state"] == "SUCCESS"
        assert body["result"] is not None
        assert body["result"]["chapters_processed"] == 2
        assert body["result"]["chunks_synthesized"] >= 2
        assert body["result"]["output_size_bytes"] > 0
        assert body["result"]["output_path"].endswith(".mp3")
        # Plik faktycznie istnieje
        assert Path(body["result"]["output_path"]).is_file()

    def test_status_nieznanego_taska_zwraca_PENDING(self, client: TestClient):
        r = client.get("/status/nie-istnieje-taki-task-id")
        assert r.status_code == 200
        body = r.json()
        assert body["state"] == "PENDING"
        assert body["result"] is None
        assert body["error"] is None


# ============================================================
#  Status - tryb FAILURE
# ============================================================


class TestStatusFailure:
    def test_uszkodzony_epub_konczy_zadanie_jako_FAILURE(
        self, client: TestClient
    ):
        # Wgrywamy "EPUB" który jest bzdurnymi bajtami - EbookLib wybuchnie.
        garbage = b"to nie jest EPUB" * 100
        upload = client.post(
            "/upload",
            files={"file": ("zly.epub", garbage, "application/epub+zip")},
        )
        assert upload.status_code == 202
        task_id = upload.json()["task_id"]

        status = client.get(f"/status/{task_id}")
        assert status.status_code == 200
        body = status.json()
        assert body["state"] == "FAILURE"
        assert body["error"] is not None
        assert len(body["error"]) > 0


# ============================================================
#  Download endpoint
# ============================================================


class TestDownloadHappyPath:
    def test_po_sukcesie_zwraca_mp3(
        self, client: TestClient, epub_payload: bytes
    ):
        # 1. Upload + task (eager - kończy się natychmiast)
        upload = client.post(
            "/upload",
            files={"file": ("k.epub", epub_payload, "application/epub+zip")},
        )
        task_id = upload.json()["task_id"]

        # 2. Download
        r = client.get(f"/download/{task_id}")
        assert r.status_code == 200
        assert r.headers["content-type"] == "audio/mpeg"
        assert 'attachment' in r.headers.get("content-disposition", "")
        assert ".mp3" in r.headers.get("content-disposition", "")
        # Zawartość: 2 bajty mp3 header (ID3 albo MPEG sync frame)
        head = r.content[:4]
        is_id3 = head[:3] == b"ID3"
        is_mpeg = head[0] == 0xFF and (head[1] & 0xE0) == 0xE0
        assert is_id3 or is_mpeg


class TestDownloadStanyTaska:
    def test_pending_zwraca_404(self, client: TestClient):
        # Poprawny UUID, ale task nie istnieje
        fake_id = "12345678-1234-1234-1234-123456789012"
        r = client.get(f"/download/{fake_id}")
        assert r.status_code == 404
        assert "nie istnieje" in r.json()["detail"].lower()

    def test_failure_zwraca_500(self, client: TestClient):
        # Wgrywamy śmieci - task pada
        upload = client.post(
            "/upload",
            files={"file": ("zly.epub", b"smieci" * 50, "application/epub+zip")},
        )
        task_id = upload.json()["task_id"]

        r = client.get(f"/download/{task_id}")
        assert r.status_code == 500
        assert "błędem" in r.json()["detail"].lower()


class TestDownloadBezpieczenstwo:
    @pytest.mark.parametrize(
        "bad_id",
        [
            "../../../etc/passwd",
            "../etc/passwd",
            "not-a-uuid",
            "12345",
            "12345678_1234_1234_1234_123456789012",  # podkreślniki zamiast minusów
            "12345678-1234-1234-1234-12345678901",   # za krótkie
            "12345678-1234-1234-1234-1234567890123", # za długie
            "GGGGGGGG-1234-1234-1234-123456789012",  # nie-hex
            "",
        ],
    )
    def test_zly_format_task_id_zwraca_400(
        self, client: TestClient, bad_id: str
    ):
        r = client.get(f"/download/{bad_id}")
        # Pusty bad_id daje 404 (route not matched), reszta - 400 albo 404.
        # Kluczowe: NIGDY 200, NIGDY treść spoza output_dir.
        assert r.status_code in (400, 404)

    def test_payload_z_plikiem_spoza_output_dir_zwraca_403(
        self,
        client: TestClient,
        test_settings: Settings,
        tmp_path: Path,
    ):
        """
        Symulujemy stan, w którym task SUCCESS zwrócił ścieżkę POZA
        settings.output_dir (np. atak przez manipulację Redisa).
        Endpoint MUSI zwrócić 403 i NIE może oddać tego pliku.
        """
        # Tworzymy "tajny" plik POZA output_dir
        secret = tmp_path / "secret.mp3"
        secret.write_bytes(b"ID3\x00\x00\x00\x00\x00sekrety")

        fake_task_id = "deadbeef-1234-5678-9abc-def012345678"

        # Wstrzykujemy do backendu Celery payload wskazujący na ten plik
        from celery.backends.base import DisabledBackend
        backend = celery_app.backend
        if isinstance(backend, DisabledBackend):
            pytest.skip("Backend wyłączony - nie da się wstrzyknąć stanu.")
        backend.store_result(
            fake_task_id,
            {
                "output_path": str(secret),  # POZA test_settings.output_dir
                "chapters_processed": 1,
                "chunks_synthesized": 1,
                "output_size_bytes": secret.stat().st_size,
            },
            "SUCCESS",
        )

        r = client.get(f"/download/{fake_task_id}")
        assert r.status_code == 403
        assert "poza" in r.json()["detail"].lower()
        # Plik dalej istnieje - endpoint go nawet nie otworzył.
        assert secret.exists()

    def test_payload_wskazuje_na_nieistniejacy_plik_zwraca_404(
        self,
        client: TestClient,
        test_settings: Settings,
    ):
        """Task SUCCESS, ale plik został w międzyczasie usunięty z dysku."""
        fake_path = test_settings.output_dir / "znikla.mp3"
        # NIE tworzymy pliku
        fake_task_id = "cafebabe-1234-5678-9abc-def012345678"

        from celery.backends.base import DisabledBackend
        backend = celery_app.backend
        if isinstance(backend, DisabledBackend):
            pytest.skip("Backend wyłączony.")
        test_settings.output_dir.mkdir(parents=True, exist_ok=True)
        backend.store_result(
            fake_task_id,
            {
                "output_path": str(fake_path),
                "chapters_processed": 1,
                "chunks_synthesized": 1,
                "output_size_bytes": 0,
            },
            "SUCCESS",
        )

        r = client.get(f"/download/{fake_task_id}")
        assert r.status_code == 404
        assert "nie istnieje" in r.json()["detail"].lower()


# ============================================================
#  Queue endpoint - widok kolejki
# ============================================================


class TestQueueEndpoint:
    """
    Testy /queue z mockowanym `inspect()`. Nie potrzebujemy realnego
    brokera ani workera - patchujemy `celery_app.control.inspect`.
    """

    def test_brak_workerow_zwraca_pusta_liste(
        self, client: TestClient, monkeypatch
    ):
        """Broker działa, ale żaden worker nie odpowiada na ping."""
        from app.worker import celery_app

        class _FakeInspector:
            def ping(self): return {}        # zero workerów
            def active(self): return {}
            def reserved(self): return {}

        monkeypatch.setattr(
            celery_app.control, "inspect",
            lambda *a, **kw: _FakeInspector(),
        )

        r = client.get("/queue")
        assert r.status_code == 200
        body = r.json()
        assert body["items"] == []
        assert body["workers_online"] == 0
        assert body["broker_reachable"] is True

    def test_active_i_reserved_taski_zwracane_lacznie(
        self, client: TestClient, monkeypatch
    ):
        from app.worker import celery_app

        class _FakeInspector:
            def ping(self):
                return {"celery@worker1": {"ok": "pong"}}
            def active(self):
                return {
                    "celery@worker1": [
                        {
                            "id": "aaa-111",
                            "name": "audio_ksiaznica.process_epub",
                            "args": ["/storage/uploads/uuid__moja_ksiazka.epub"],
                            "time_start": 1700000000.5,
                        }
                    ]
                }
            def reserved(self):
                return {
                    "celery@worker1": [
                        {
                            "id": "bbb-222",
                            "name": "audio_ksiaznica.process_epub",
                            "args": ["/storage/uploads/uuid__druga.epub"],
                            "time_start": None,
                        }
                    ]
                }

        monkeypatch.setattr(
            celery_app.control, "inspect",
            lambda *a, **kw: _FakeInspector(),
        )

        r = client.get("/queue")
        assert r.status_code == 200
        body = r.json()
        assert body["workers_online"] == 1
        assert body["broker_reachable"] is True
        assert len(body["items"]) == 2

        # ACTIVE musi być przed RESERVED
        states = [item["state"] for item in body["items"]]
        assert states == ["ACTIVE", "RESERVED"]

        # Wydedukowane nazwy plików
        filenames = [item["epub_filename"] for item in body["items"]]
        assert "moja_ksiazka.epub" in filenames
        assert "druga.epub" in filenames

    def test_broker_niedostepny_zwraca_503(
        self, client: TestClient, monkeypatch
    ):
        from app.worker import celery_app

        class _DeadInspector:
            def ping(self):
                raise ConnectionError("Connection refused (Redis down)")

        monkeypatch.setattr(
            celery_app.control, "inspect",
            lambda *a, **kw: _DeadInspector(),
        )

        r = client.get("/queue")
        assert r.status_code == 503
        assert "broker" in r.json()["detail"].lower()

    def test_nazwa_pliku_bez_prefiksu_uuid(
        self, client: TestClient, monkeypatch
    ):
        """Jeśli args nie ma UUID-prefiksu, zwracamy całą nazwę."""
        from app.worker import celery_app

        class _FakeInspector:
            def ping(self):
                return {"celery@w1": {"ok": "pong"}}
            def active(self):
                return {
                    "celery@w1": [
                        {
                            "id": "z-1",
                            "name": "audio_ksiaznica.process_epub",
                            "args": ["/somewhere/legacy_name.epub"],
                        }
                    ]
                }
            def reserved(self): return {}

        monkeypatch.setattr(
            celery_app.control, "inspect",
            lambda *a, **kw: _FakeInspector(),
        )

        r = client.get("/queue")
        body = r.json()
        assert body["items"][0]["epub_filename"] == "legacy_name.epub"

    def test_uszkodzone_args_nie_psuja_endpointu(
        self, client: TestClient, monkeypatch
    ):
        """Defensywnie - jak args jest pusty/None/dziwnego typu, nie pada."""
        from app.worker import celery_app

        class _FakeInspector:
            def ping(self):
                return {"celery@w1": {"ok": "pong"}}
            def active(self):
                return {
                    "celery@w1": [
                        {"id": "1", "name": "task", "args": []},
                        {"id": "2", "name": "task", "args": None},
                        {"id": "3", "name": "task", "args": [None]},
                        {"id": "4", "name": "task"},  # bez args
                    ]
                }
            def reserved(self): return {}

        monkeypatch.setattr(
            celery_app.control, "inspect",
            lambda *a, **kw: _FakeInspector(),
        )

        r = client.get("/queue")
        assert r.status_code == 200
        body = r.json()
        assert len(body["items"]) == 4
        for item in body["items"]:
            assert item["epub_filename"] is None
