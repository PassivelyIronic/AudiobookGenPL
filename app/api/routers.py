"""
Endpointy FastAPI.

Dwa endpointy:
    POST /upload           - przyjmuje EPUB, kolejkuje task, zwraca task_id
    GET  /status/{task_id} - czyta stan z Celery result backend

Streaming uploadu:
    Plik jest zapisywany przez `aiofiles` w kawałkach po 1 MB - nie ląduje
    cały w RAM-ie, więc 50 MB EPUB nie zje serwera. Po każdym kawałku
    sprawdzamy łączny rozmiar - przy przekroczeniu limitu kasujemy plik
    i zwracamy 413.

Bezpieczeństwo:
    * walidujemy rozszerzenie pliku,
    * używamy Path(file.filename).name żeby uciąć ewentualny path traversal
      ('../../../etc/passwd'),
    * UUID jako prefix nazwy - dwa upload'y o tej samej nazwie się nie biją.
"""
from __future__ import annotations

import logging
import re
import uuid
from pathlib import Path

import aiofiles
from celery.result import AsyncResult
from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    UploadFile,
    status,
)
from fastapi.responses import FileResponse

from app.api.schemas import (
    ConversionResultPayload,
    HealthResponse,
    ProgressInfo,
    QueueItem,
    QueueResponse,
    TaskStatusResponse,
    UploadResponse,
)
from app.core.config import Settings, get_settings
from app.worker import celery_app, process_epub_task

logger = logging.getLogger(__name__)

router = APIRouter()


# ============================================================
#  Health
# ============================================================


@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["meta"],
    summary="Healthcheck",
)
async def health() -> HealthResponse:
    return HealthResponse()


# ============================================================
#  Upload
# ============================================================


_UPLOAD_CHUNK_SIZE = 1024 * 1024  # 1 MB


@router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["conversion"],
    summary="Wgraj EPUB i rozpocznij konwersję do MP3",
)
async def upload_epub(
    file: UploadFile = File(..., description="Plik .epub do konwersji"),
    settings: Settings = Depends(get_settings),
) -> UploadResponse:
    # ----- walidacja podstawowa -----
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Brak nazwy pliku w żądaniu.",
        )

    # Path(...).name odcina ewentualne '../' z nazwy.
    safe_name = Path(file.filename).name
    ext = Path(safe_name).suffix.lower()
    if ext not in settings.allowed_upload_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Nieobsługiwane rozszerzenie '{ext}'. "
                f"Akceptujemy: {', '.join(settings.allowed_upload_extensions)}"
            ),
        )

    # ----- przygotowanie miejsca docelowego -----
    settings.ensure_directories()
    upload_id = uuid.uuid4().hex
    target = settings.upload_dir / f"{upload_id}__{safe_name}"

    # ----- streaming na dysk z walidacją rozmiaru -----
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    written = 0
    is_oversized = False

    try:
        async with aiofiles.open(target, "wb") as out:
            while chunk := await file.read(_UPLOAD_CHUNK_SIZE):
                written += len(chunk)
                if written > max_bytes:
                    is_oversized = True
                    break  # Wychodzimy z pętli i zamykamy plik
                await out.write(chunk)
                
        # Plik jest już zamknięty, Windows pozwoli go usunąć
        if is_oversized:
            target.unlink(missing_ok=True)
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Plik przekracza limit {settings.max_upload_size_mb} MB.",
            )
    except Exception as exc:
        if not isinstance(exc, HTTPException):
            target.unlink(missing_ok=True)
            logger.exception("Błąd zapisu uploadu")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Nie udało się zapisać uploadu: {exc}",
            ) from exc
        raise  # <--- TUTAJ TYLKO SAMO "raise", żeby przepuścić błąd 413!

    if written == 0:
        target.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Otrzymano pusty plik.",
        )

    logger.info("Upload %s zapisany (%d B), kolejkuję task.", target.name, written)

    # ----- kolejkujemy task -----
    async_result = process_epub_task.delay(str(target))

    return UploadResponse(
        task_id=async_result.id,
        status_url=f"/status/{async_result.id}",
    )


# ============================================================
#  Status
# ============================================================


# Stany Celery, które nie są publiczne - mapujemy na PENDING.
_INTERNAL_STATES_AS_PENDING = {"RETRY", "REVOKED"}


@router.get(
    "/status/{task_id}",
    response_model=TaskStatusResponse,
    tags=["conversion"],
    summary="Sprawdź stan zadania konwersji",
)
async def get_task_status(task_id: str) -> TaskStatusResponse:
    async_result = AsyncResult(task_id, app=celery_app)
    raw_state = async_result.state

    # Mapowanie stanów wewnętrznych na publiczne.
    state = "PENDING" if raw_state in _INTERNAL_STATES_AS_PENDING else raw_state

    response = TaskStatusResponse(task_id=task_id, state=state)

    if raw_state == "SUCCESS":
        payload = async_result.result or {}
        try:
            response.result = ConversionResultPayload(**payload)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Task %s zwrócił dziwny payload: %s (%s)", task_id, payload, exc
            )
            response.state = "FAILURE"
            response.error = f"Niepoprawny payload taska: {exc}"

    elif raw_state == "FAILURE":
        # async_result.result / .info to wyjątek lub jego stringowa reprezentacja.
        info = async_result.info
        response.error = (
            str(info) if info is not None else "Zadanie zakończyło się błędem."
        )

    elif raw_state == "PROGRESS":
        info = async_result.info or {}
        try:
            response.progress = ProgressInfo(**info)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Task %s ma uszkodzone meta progressu: %s", task_id, exc)

    # PENDING / STARTED - nic dodatkowego nie zwracamy.
    return response


# ============================================================
#  Kolejka - widok wszystkich zadań w Celery
# ============================================================


def _extract_epub_filename(args: list) -> str | None:
    """
    Bezpiecznie wyciąga nazwę pliku z pierwszego argumentu taska.

    Format args z `process_epub_task.delay(str(target))`:
        ["/storage/uploads/abc123__moja_ksiazka.epub"]
    Zwracamy "moja_ksiazka.epub" (część po prefiksie UUID + "__").
    """
    if not args or not isinstance(args[0], str):
        return None
    try:
        name = Path(args[0]).name
        # Nasz prefix to "{uuid_hex}__{original_name}" - odcinamy uuid.
        if "__" in name:
            return name.split("__", 1)[1]
        return name
    except Exception:  # noqa: BLE001
        return None


@router.get(
    "/queue",
    response_model=QueueResponse,
    tags=["meta"],
    summary="Lista zadań w kolejce Celery",
    responses={
        200: {"description": "Aktualny stan kolejki."},
        503: {"description": "Broker Celery nieosiągalny."},
    },
)
async def list_queue() -> QueueResponse:
    """
    Zwraca aktywne (ACTIVE) i zakolejkowane (RESERVED) zadania ze wszystkich
    workerów. Wymaga, by Redis/broker były osiągalne - inaczej 503.

    Endpoint blokujący - `inspect()` wysyła broadcast pingi do workerów
    i czeka na odpowiedzi (timeout 2 s). Nie wołać częściej niż raz / 2-3 s.
    """
    inspector = celery_app.control.inspect(timeout=2.0)

    # ping() zwraca {worker_name: {'ok': 'pong'}} lub None, jeśli broker
    # nie odpowiada / brak workerów.
    try:
        ping_result = inspector.ping()
    except Exception as exc:  # noqa: BLE001 - broker może być down
        logger.warning("Nie udało się sięgnąć do brokera Celery: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Broker Celery niedostępny: {exc}",
        ) from exc

    workers_online = len(ping_result or {})

    if workers_online == 0:
        # Broker działa (ping nie rzucił), ale żaden worker nie odpowiedział.
        # Zwracamy pustą listę, NIE 503 - klient ma wiedzieć, że workery są
        # po prostu offline.
        return QueueResponse(
            items=[], workers_online=0, broker_reachable=True
        )

    active = inspector.active() or {}
    reserved = inspector.reserved() or {}

    items: list[QueueItem] = []
    for worker_name, tasks in active.items():
        for task in tasks:
            items.append(
                QueueItem(
                    task_id=task.get("id", ""),
                    state="ACTIVE",
                    worker=worker_name,
                    name=task.get("name"),
                    epub_filename=_extract_epub_filename(task.get("args") or []),
                    received_at=task.get("time_start"),
                )
            )

    for worker_name, tasks in reserved.items():
        for task in tasks:
            items.append(
                QueueItem(
                    task_id=task.get("id", ""),
                    state="RESERVED",
                    worker=worker_name,
                    name=task.get("name"),
                    epub_filename=_extract_epub_filename(task.get("args") or []),
                    received_at=task.get("time_start"),
                )
            )

    # ACTIVE przed RESERVED, w obrębie grupy - najstarsze na górze.
    items.sort(key=lambda i: (i.state != "ACTIVE", i.received_at or 0))

    return QueueResponse(
        items=items,
        workers_online=workers_online,
        broker_reachable=True,
    )


# ============================================================
#  Download
# ============================================================


# Celery generuje task_id jako UUID4 (np. "a1b2c3d4-e5f6-7890-abcd-ef1234567890").
# Restrykcyjny regex blokuje próby path-traversalu (`../`, znaki specjalne)
# zanim w ogóle dotkniemy backendu Celery.
_TASK_ID_RE: re.Pattern[str] = re.compile(
    r"^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$",
    re.IGNORECASE,
)


@router.get(
    "/download/{task_id}",
    tags=["conversion"],
    summary="Pobierz wygenerowany audiobook MP3",
    response_class=FileResponse,
    responses={
        200: {
            "content": {"audio/mpeg": {}},
            "description": "Plik MP3 gotowy do pobrania.",
        },
        400: {"description": "Niepoprawny format task_id."},
        403: {"description": "Plik poza dozwolonym katalogiem (path traversal)."},
        404: {"description": "Zadanie nie istnieje lub plik został usunięty."},
        425: {"description": "Zadanie jeszcze się nie zakończyło."},
        500: {"description": "Zadanie zakończone błędem."},
    },
)
async def download_mp3(
    task_id: str,
    settings: Settings = Depends(get_settings),
) -> FileResponse:
    """
    Zwraca plik MP3 wygenerowany przez `process_epub_task`.

    Etapy weryfikacji (każdy z osobnym kodem HTTP):
        1. task_id musi być poprawnym UUID4 - inaczej 400.
        2. Stan taska MUSI być SUCCESS - inaczej 404/425/500.
        3. Ścieżka pliku z payloadu MUSI być wewnątrz `settings.output_dir`
           (ochrona przed manipulacją wyniku w backendzie) - inaczej 403.
        4. Plik MUSI istnieć na dysku - inaczej 404.
    """
    # ----- 1. Walidacja formatu task_id (anti-path-traversal) -----
    if not _TASK_ID_RE.match(task_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Niepoprawny format task_id - oczekiwany UUID4.",
        )

    # ----- 2. Sprawdzenie stanu taska -----
    async_result = AsyncResult(task_id, app=celery_app)
    state = async_result.state

    if state == "PENDING":
        # PENDING w Celery oznacza także "task w ogóle nie istnieje" -
        # backend nie odróżnia. 404 jest dla klienta jaśniejsze niż "czekaj".
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Zadanie nie istnieje albo jeszcze nie wystartowało.",
        )

    if state in ("STARTED", "PROGRESS", "RETRY"):
        raise HTTPException(
            status_code=status.HTTP_425_TOO_EARLY,
            detail=f"Zadanie wciąż przetwarza (stan: {state}). Spróbuj później.",
        )

    if state == "FAILURE":
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Zadanie zakończyło się błędem - pobierz status z /status/{task_id}.",
        )

    if state != "SUCCESS":
        # REVOKED i inne dziwne stany
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Nieobsługiwany stan zadania: {state}",
        )

    # ----- 3. Walidacja payloadu i ścieżki -----
    payload = async_result.result or {}
    raw_path = payload.get("output_path")
    if not raw_path:
        logger.error("Task %s ma stan SUCCESS, ale brak output_path w payload", task_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Zadanie SUCCESS, ale wynik nie zawiera ścieżki pliku.",
        )

    output_path = Path(raw_path).resolve()
    output_dir = settings.output_dir.resolve()

    # Path.relative_to rzuca ValueError, jeśli `output_path` nie jest
    # potomkiem `output_dir`. To NASZ ostatni mur obronny - gdyby ktoś
    # zmanipulował payload w Redisie i wstawił "/etc/passwd", tu go łapiemy.
    try:
        output_path.relative_to(output_dir)
    except ValueError:
        logger.warning(
            "Odrzucam próbę pobrania pliku spoza output_dir: %s (dozwolone: %s)",
            output_path,
            output_dir,
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Plik znajduje się poza dozwolonym katalogiem.",
        )

    # ----- 4. Plik na dysku -----
    if not output_path.is_file():
        logger.warning("Plik MP3 dla taska %s nie istnieje: %s", task_id, output_path)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Plik został wygenerowany, ale już nie istnieje na dysku.",
        )

    # ----- 5. Zwracamy ZE STREAMINGIEM (FileResponse robi sendfile) -----
    # FileResponse używa sendfile(2) pod spodem - plik leci z dysku do klienta
    # *bez* ładowania do RAM. Dla 200 MB audiobooka to ratunek dla pamięci.
    return FileResponse(
        path=output_path,
        media_type="audio/mpeg",
        filename=output_path.name,
        headers={
            "Content-Disposition": f'attachment; filename="{output_path.name}"',
        },
    )
