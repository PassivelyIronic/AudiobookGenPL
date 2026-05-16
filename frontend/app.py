"""
Audio-Książnica — Gradio UI.

Frontend jako w pełni niezależny klient FastAPI - komunikuje się tylko
przez HTTP (httpx), więc działa też w docker-compose z osobnym kontenerem
backendu, byleby URL `AK_API_URL` wskazywał na właściwego hosta.

Uruchomienie:
    # 1. Postaw backend (Redis + worker + FastAPI - osobno, jak w fazie 2)
    # 2. Odpal UI:
    python frontend/app.py
    # → http://localhost:7860

Konfiguracja przez zmienne środowiskowe (opcjonalne):
    AK_API_URL         - adres FastAPI (domyślnie http://localhost:8000)
    AK_UI_HOST         - bind hosta dla Gradio (domyślnie 0.0.0.0)
    AK_UI_PORT         - port dla Gradio (domyślnie 7860)
    AK_UI_POLL_SEC     - interwał pollingu /status w sekundach (domyślnie 2.0)
"""
from __future__ import annotations

import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Generator

import gradio as gr
import httpx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("audio-ksiaznica-ui")


# ============================================================
#  Konfiguracja
# ============================================================


DEFAULT_API_URL: str = os.environ.get("AK_API_URL", "http://localhost:8000")
UI_HOST: str = os.environ.get("AK_UI_HOST", "0.0.0.0")
UI_PORT: int = int(os.environ.get("AK_UI_PORT", "7860"))
POLL_INTERVAL_SEC: float = float(os.environ.get("AK_UI_POLL_SEC", "2.0"))

# Twardy timeout pollingu - 6h. Po tym czasie UI rezygnuje, task w workerze
# może dalej pracować, ale klient już nie czeka.
POLL_TIMEOUT_SEC: int = 3600 * 6

UPLOAD_HTTP_TIMEOUT: float = 60.0
STATUS_HTTP_TIMEOUT: float = 10.0
DOWNLOAD_HTTP_TIMEOUT: float = 600.0
DOWNLOAD_CHUNK_SIZE: int = 1024 * 1024  # 1 MB

# Czytelne etykiety etapów - mapują techniczne nazwy z pipeline'u na UI.
STAGE_LABELS: dict[str, str] = {
    "parsing": "📖 Parsuję EPUB",
    "synthesizing": "🎤 Syntezuję mowę",
    "stitching": "🎵 Łączę audio do MP3",
    "done": "✅ Zakończono",
}

# Format HTTP-error → emoji w UI
ERROR_EMOJI: dict[int, str] = {
    400: "🚫",
    403: "🔒",
    404: "❓",
    413: "📦",
    422: "📝",
    425: "⏳",
    500: "💥",
}


# ============================================================
#  Funkcje pomocnicze - komunikacja z API
# ============================================================


def _format_api_error(response: httpx.Response) -> str:
    """Buduje czytelny markdown z błędu HTTP."""
    code = response.status_code
    emoji = ERROR_EMOJI.get(code, "❌")
    try:
        detail = response.json().get("detail", response.text)
    except Exception:
        detail = (response.text or "Brak szczegółów")[:500]
    return (
        f"### {emoji} Błąd API (HTTP {code})\n\n"
        f"```\n{detail}\n```"
    )


def _format_progress_md(task_id: str, body: dict[str, Any]) -> str:
    """Formatuje stan PROGRESS w czytelny markdown."""
    p = body.get("progress") or {}
    stage = p.get("stage", "?")
    label = STAGE_LABELS.get(stage, f"⚙️ {stage}")
    cur = p.get("current", 0)
    tot = p.get("total", 0)
    percent = p.get("percent", 0.0)
    msg = p.get("message", "")

    counter = f"{cur} / {tot}" if tot else f"{cur}"
    return (
        f"### {label}\n\n"
        f"- **Postęp:** {counter} ({percent:.1f}%)\n"
        f"- **Aktualnie:** {msg}\n\n"
        f"<sub>Task: `{task_id}`</sub>"
    )


def _format_success_md(task_id: str, result: dict[str, Any]) -> str:
    chapters = result.get("chapters_processed", "?")
    chunks = result.get("chunks_synthesized", "?")
    size_mb = result.get("output_size_bytes", 0) / 1024 / 1024
    name = Path(result.get("output_path", "audiobook.mp3")).name
    return (
        f"### ✅ Audiobook gotowy!\n\n"
        f"| | |\n"
        f"|---|---|\n"
        f"| 📚 Rozdziały | **{chapters}** |\n"
        f"| 🎤 Chunki audio | **{chunks}** |\n"
        f"| 💾 Rozmiar | **{size_mb:.1f} MB** |\n"
        f"| 📁 Plik | `{name}` |\n\n"
        f"<sub>Task: `{task_id}`</sub>"
    )


def _download_mp3_streaming(api_url: str, task_id: str) -> Path | None:
    """
    Strumieniowo pobiera MP3 z `/download/{task_id}` do pliku tymczasowego.

    NIE ładuje pliku do RAM - dla 200 MB audiobooka to ma znaczenie.
    """
    target = Path(tempfile.gettempdir()) / f"audiobook_{task_id}.mp3"
    try:
        with httpx.stream(
            "GET",
            f"{api_url}/download/{task_id}",
            timeout=DOWNLOAD_HTTP_TIMEOUT,
            follow_redirects=True,
        ) as r:
            if r.status_code != 200:
                logger.error(
                    "Download failed: HTTP %s (%s)",
                    r.status_code,
                    r.read()[:200],
                )
                return None
            with target.open("wb") as fh:
                for chunk in r.iter_bytes(chunk_size=DOWNLOAD_CHUNK_SIZE):
                    fh.write(chunk)
        return target
    except httpx.RequestError as exc:
        logger.exception("Błąd przy pobieraniu MP3: %s", exc)
        return None


# ============================================================
#  Główna funkcja konwersji - generator yieldujący updaty
# ============================================================


def convert_epub(
    file_path: str | None,
    api_url: str,
    progress: gr.Progress = gr.Progress(),
) -> Generator[tuple[Any, str], None, None]:
    """
    Wgrywa EPUB do API, pollinguje status i zwraca gotowe MP3.

    Każdy `yield` aktualizuje 2 outputy: (audio_path, status_markdown).
    `audio_path=None` znaczy "nie pokazuj audio jeszcze".
    """
    # ----- 0. Walidacja wejścia -----
    if not file_path:
        yield None, "⚠️ **Wgraj plik EPUB przed rozpoczęciem.**"
        return

    epub_path = Path(file_path)
    if not epub_path.is_file():
        yield None, f"⚠️ **Plik nie istnieje:** `{epub_path}`"
        return

    api_url = (api_url or DEFAULT_API_URL).rstrip("/")

    # ----- 1. Upload -----
    progress(0.02, desc="Wgrywam EPUB")
    yield None, f"📤 **Wgrywam `{epub_path.name}` do {api_url}...**"

    try:
        with epub_path.open("rb") as fh:
            resp = httpx.post(
                f"{api_url}/upload",
                files={
                    "file": (
                        epub_path.name,
                        fh,
                        "application/epub+zip",
                    )
                },
                timeout=UPLOAD_HTTP_TIMEOUT,
            )
    except httpx.RequestError as exc:
        yield None, (
            f"❌ **Brak połączenia z API** (`{api_url}`)\n\n"
            f"```\n{exc}\n```\n\n"
            f"Sprawdź, czy backend Audio-Książnicy działa."
        )
        return

    if resp.status_code != 202:
        yield None, _format_api_error(resp)
        return

    task_id = resp.json()["task_id"]
    progress(0.05, desc="Task zakolejkowany")
    yield None, f"📋 **Task zakolejkowany:** `{task_id}`\n\nCzekam na workera..."

    # ----- 2. Polling /status -----
    start = time.monotonic()
    last_state = None

    while True:
        if time.monotonic() - start > POLL_TIMEOUT_SEC:
            yield None, (
                f"⏱️ **Timeout** — przekroczono {POLL_TIMEOUT_SEC // 3600}h "
                f"oczekiwania.\n\n"
                f"Task `{task_id}` może wciąż działać w workerze."
            )
            return

        try:
            r = httpx.get(
                f"{api_url}/status/{task_id}",
                timeout=STATUS_HTTP_TIMEOUT,
            )
        except httpx.RequestError as exc:
            logger.warning("Polling /status zawiódł: %s — retry...", exc)
            time.sleep(POLL_INTERVAL_SEC)
            continue

        if r.status_code != 200:
            yield None, _format_api_error(r)
            return

        body = r.json()
        state = body["state"]

        # --- SUCCESS: pobieramy plik i kończymy ---
        if state == "SUCCESS":
            progress(0.95, desc="Pobieram MP3")
            yield None, "📥 **Pobieram gotowy plik MP3 z API...**"

            mp3 = _download_mp3_streaming(api_url, task_id)
            if mp3 is None:
                yield None, (
                    "⚠️ **Konwersja udana, ale plik niedostępny.**\n\n"
                    f"Spróbuj otworzyć: `{api_url}/download/{task_id}`"
                )
                return

            progress(1.0, desc="Gotowe")
            yield str(mp3), _format_success_md(task_id, body.get("result") or {})
            return

        # --- FAILURE: pokazujemy błąd ---
        if state == "FAILURE":
            err = body.get("error") or "Nieznany błąd (brak szczegółów)."
            yield None, (
                f"### ❌ Konwersja nie powiodła się\n\n"
                f"```\n{err}\n```\n\n"
                f"<sub>Task: `{task_id}`</sub>"
            )
            return

        # --- PROGRESS / STARTED / PENDING: aktualizujemy UI ---
        if state == "PROGRESS":
            percent = (body.get("progress") or {}).get("percent", 0.0) / 100.0
            stage = (body.get("progress") or {}).get("stage", "?")
            label = STAGE_LABELS.get(stage, stage)
            # Mapujemy postęp pipeline'u na zakres 0.10 - 0.90 paska
            # (pozostawiamy bufor na upload i download).
            progress(0.10 + percent * 0.80, desc=label)
            yield None, _format_progress_md(task_id, body)

        elif state == "STARTED":
            if state != last_state:
                progress(0.08, desc="Startuję workera")
                yield None, "🚀 **Worker rozpoczął przetwarzanie...**"

        elif state == "PENDING":
            if state != last_state:
                yield None, "⏳ **Zadanie w kolejce, czekam na workera...**"

        else:
            yield None, f"ℹ️ Nieznany stan: `{state}`"

        last_state = state
        time.sleep(POLL_INTERVAL_SEC)


# ============================================================
#  Reset UI - czyści wszystkie outputy między konwersjami
# ============================================================


def reset_ui() -> tuple[None, None, str]:
    return None, None, "👋 **Wgraj nowy plik EPUB i kliknij \"Konwertuj\".**"


# ============================================================
#  Kolejka - pobieranie stanu z /queue
# ============================================================


def _format_received_at(ts: float | None) -> str:
    """Unix timestamp -> 'HH:MM:SS' (lokalny czas) albo '-'."""
    if not ts:
        return "—"
    try:
        from datetime import datetime
        return datetime.fromtimestamp(ts).strftime("%H:%M:%S")
    except Exception:  # noqa: BLE001
        return "—"


# Etykiety stanów dla UI - emoji + nazwa
_QUEUE_STATE_LABELS: dict[str, str] = {
    "ACTIVE": "🟢 Przetwarzanie",
    "RESERVED": "⏳ W kolejce",
}


def fetch_queue(api_url: str) -> tuple[list[list[str]], str]:
    """
    Odpytuje /queue i zwraca dane do wyświetlenia w Gradio Dataframe.

    Returns:
        (rows, summary_markdown) - rows to lista wierszy dla gr.Dataframe,
        summary_markdown to krótki opis nad tabelą.
    """
    api_url = (api_url or DEFAULT_API_URL).rstrip("/")

    try:
        r = httpx.get(f"{api_url}/queue", timeout=5.0)
    except httpx.RequestError as exc:
        return (
            [],
            f"❌ **Brak połączenia z API** (`{api_url}`)\n\n```\n{exc}\n```",
        )

    if r.status_code == 503:
        try:
            detail = r.json().get("detail", "broker offline")
        except Exception:
            detail = "broker offline"
        return [], f"⚠️ **Broker Celery niedostępny:** {detail}"

    if r.status_code != 200:
        return [], _format_api_error(r)

    body = r.json()
    workers = body.get("workers_online", 0)
    items = body.get("items", [])

    if workers == 0:
        summary = (
            "💤 **Brak workerów online.** Uruchom `celery -A app.worker.celery_app worker`."
        )
    elif not items:
        summary = (
            f"😴 **Kolejka pusta** — {workers} worker(ów) gotowych na zadania."
        )
    else:
        active = sum(1 for i in items if i["state"] == "ACTIVE")
        reserved = sum(1 for i in items if i["state"] == "RESERVED")
        summary = (
            f"### 📋 W kolejce: **{len(items)}** zadań  \n"
            f"🟢 Przetwarzanych: **{active}** · "
            f"⏳ Oczekujących: **{reserved}** · "
            f"👷 Workerów online: **{workers}**"
        )

    rows = []
    for item in items:
        rows.append([
            _QUEUE_STATE_LABELS.get(item["state"], item["state"]),
            item.get("epub_filename") or "—",
            item["task_id"][:8] + "…",
            item.get("worker") or "—",
            _format_received_at(item.get("received_at")),
        ])
    return rows, summary


# ============================================================
#  Layout
# ============================================================


_DESCRIPTION_MD = """
**Audio-Książnica** konwertuje pliki EPUB na audiobooki MP3 z polskim
głosem (XTTS-v2). Backend jest asynchroniczny — wgrywasz plik, dostajesz
identyfikator zadania, a UI pokazuje postęp na żywo.

> ℹ️ Backend musi być uruchomiony osobno (Redis + Celery worker + FastAPI).
> Zobacz `docker-compose.yml` i instrukcje w README.
"""


def build_ui() -> gr.Blocks:
    with gr.Blocks(
        title="Audio-Książnica",
        analytics_enabled=False,
    ) as demo:
        gr.Markdown("# 📚 Audio-Książnica")
        gr.Markdown(_DESCRIPTION_MD)

        # URL backendu jest WSPÓLNY dla obu zakładek - trzymamy go nad
        # zakładkami, żeby zmiana w jednym miejscu propagowała się
        # do drugiej zakładki.
        api_url_input = gr.Textbox(
            label="URL backendu FastAPI",
            value=DEFAULT_API_URL,
            placeholder="http://localhost:8000",
            info="Adres działającej instancji backendu (Redis + Celery + FastAPI).",
        )

        with gr.Tabs():
            # ============================================================
            #  Zakładka 1: Konwersja
            # ============================================================
            with gr.Tab("🎙️ Konwersja"):
                with gr.Row(equal_height=False):
                    # ----- Kolumna konfiguracji -----
                    with gr.Column(scale=2):
                        gr.Markdown("### 📥 Wgraj plik")
                        file_input = gr.File(
                            label="Plik EPUB",
                            file_types=[".epub"],
                            type="filepath",
                            file_count="single",
                        )
                        with gr.Row():
                            convert_btn = gr.Button(
                                "🎙️ Konwertuj do MP3",
                                variant="primary",
                                size="lg",
                                scale=2,
                            )
                            reset_btn = gr.Button("🧹 Wyczyść", size="lg", scale=1)

                    # ----- Kolumna wyniku -----
                    with gr.Column(scale=3):
                        gr.Markdown("### 📡 Status konwersji")
                        status_md = gr.Markdown(
                            "👋 **Witaj!** Wgraj plik EPUB i kliknij \"Konwertuj\".",
                        )
                        gr.Markdown("### 🎧 Audiobook")
                        audio_output = gr.Audio(
                            label="Wynik konwersji",
                            type="filepath",
                            interactive=False,
                        )

            # ============================================================
            #  Zakładka 2: Kolejka - widok wszystkich tasków
            # ============================================================
            with gr.Tab("📋 Kolejka"):
                gr.Markdown(
                    "Lista aktywnie przetwarzanych i oczekujących zadań. "
                    "Odświeża się automatycznie co 5 sekund."
                )
                queue_summary = gr.Markdown(
                    "⏳ Ładowanie...",
                )
                queue_table = gr.Dataframe(
                    headers=["Status", "Plik", "Task ID", "Worker", "Start"],
                    datatype=["str", "str", "str", "str", "str"],
                    interactive=False,
                    wrap=True,
                    label="Zadania",
                )
                with gr.Row():
                    refresh_btn = gr.Button("🔄 Odśwież ręcznie", size="sm")

                # Auto-refresh co 5 s przez Timer + ręczny refresh przyciskiem.
                queue_timer = gr.Timer(value=5.0, active=True)
                queue_timer.tick(
                    fn=fetch_queue,
                    inputs=[api_url_input],
                    outputs=[queue_table, queue_summary],
                )
                refresh_btn.click(
                    fn=fetch_queue,
                    inputs=[api_url_input],
                    outputs=[queue_table, queue_summary],
                )
                # Pierwsze załadowanie - przy starcie demo, nie na timer.
                demo.load(
                    fn=fetch_queue,
                    inputs=[api_url_input],
                    outputs=[queue_table, queue_summary],
                )

        gr.Markdown(
            "<sub>Wskazówka: jeśli planujesz wrzucić kilka książek pod rząd, "
            "kolejka pokaże ci status każdej po kolei. "
            "Worker przetwarza po jednym tasku naraz (concurrency=1).</sub>"
        )

        # ----- Event bindings dla zakładki Konwersja -----
        convert_btn.click(
            fn=convert_epub,
            inputs=[file_input, api_url_input],
            outputs=[audio_output, status_md],
            show_progress="full",
        )
        reset_btn.click(
            fn=reset_ui,
            outputs=[file_input, audio_output, status_md],
        )

    return demo


# ============================================================
#  Entrypoint
# ============================================================


if __name__ == "__main__":
    demo = build_ui()
    # .queue() obsługuje generatory długo-żyjące (długi polling).
    demo.queue(default_concurrency_limit=4).launch(
        server_name=UI_HOST,
        server_port=UI_PORT,
        show_error=True,
        inbrowser=False,
        theme=gr.themes.Soft(
            primary_hue="indigo",
            secondary_hue="violet",
            font=["Inter", "system-ui", "sans-serif"],
        ),
    )
