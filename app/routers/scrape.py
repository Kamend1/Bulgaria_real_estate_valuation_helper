import asyncio
import json
import subprocess
import sys
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.config import settings
from app.db.models import ScrapeRun
from app.db.session import get_db, db_session
from app.services.scrape_service import get_scrape_status
from app.templating import templates

router = APIRouter(prefix="/scrape", tags=["scrape"])

_PROJECT_ROOT = Path(__file__).parent.parent.parent


def _launch_scrape(run_id: str) -> None:
    """Start run_scrape.py as an independent OS process."""
    subprocess.Popen(
        [sys.executable, "-m", "scripts.run_scrape", "--run-id", run_id],
        cwd=str(_PROJECT_ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        # On Windows, CREATE_NEW_PROCESS_GROUP ensures child survives parent death
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
    )


@router.get("/", response_class=HTMLResponse)
async def scrape_page(request: Request, db: Session = Depends(get_db)):
    recent_runs = (
        db.query(ScrapeRun)
        .order_by(ScrapeRun.started_at.desc())
        .limit(20)
        .all()
    )
    scrape_status = get_scrape_status(db)
    return templates.TemplateResponse(request, "scrape.html", {
        "recent_runs": recent_runs,
        "scrape_status": scrape_status,
    })


@router.post("/start", response_class=HTMLResponse)
async def start_scrape(
    request: Request,
    db: Session = Depends(get_db),
    deal_types: list[str] = Form(default=[]),
    geo_paths: list[str] = Form(default=[]),
    property_types: list[str] = Form(default=[]),
):
    active = (
        db.query(ScrapeRun)
        .filter(ScrapeRun.status.in_(["running", "starting"]))
        .first()
    )
    if active:
        active_id = str(active.id)
        return HTMLResponse(
            f'<div class="card" style="border-left:4px solid #f59e0b;padding:1rem">'
            f'<strong>⚠ Засичането вече е активно.</strong> '
            f'<a href="/scrape/progress_view/{active_id}">Виж прогреса</a>'
            f"</div>",
            status_code=200,
        )

    if not deal_types:
        deal_types = ["prodazhbi", "naemi"]

    run_id = str(uuid.uuid4())
    db_run = ScrapeRun(
        id=uuid.UUID(run_id),
        status="starting",
        deal_types=deal_types,
        geo_paths=geo_paths,
        property_types=property_types,
        phase="init",
    )
    db.add(db_run)
    db.commit()

    _launch_scrape(run_id)

    return templates.TemplateResponse(
        request, "scrape_progress.html", {"run_id": run_id}
    )


@router.post("/resume/{run_id}", response_class=HTMLResponse)
async def resume_scrape(
    request: Request,
    run_id: str,
    db: Session = Depends(get_db),
):
    """Resume an interrupted scrape run from its existing checkpoint files."""
    run = db.get(ScrapeRun, uuid.UUID(run_id))
    if run is None:
        return HTMLResponse("Run not found", status_code=404)

    if run.pid:
        _kill_process(run.pid)

    run.status = "running"
    run.phase = "init"
    run.stop_requested = False
    run.finished_at = None
    run.last_message = "Възстановяване…"
    run.last_heartbeat_at = datetime.now(timezone.utc)
    db.commit()

    _launch_scrape(run_id)

    return RedirectResponse(url=f"/scrape/progress_view/{run_id}", status_code=303)


@router.get("/progress_view/{run_id}", response_class=HTMLResponse)
async def progress_view(request: Request, run_id: str, db: Session = Depends(get_db)):
    """Full page that wraps the SSE progress panel — used for resume and direct links."""
    run = db.get(ScrapeRun, uuid.UUID(run_id))
    return templates.TemplateResponse(
        request, "scrape_progress_page.html", {"run_id": run_id, "run": run}
    )


@router.post("/stop/{run_id}")
async def stop_scrape(run_id: str, db: Session = Depends(get_db)):
    """Kill the scrape process by PID and mark as interrupted."""
    run_uuid = uuid.UUID(run_id)
    run = db.get(ScrapeRun, run_uuid)

    if run and run.status in ("running", "starting"):
        run.stop_requested = True
        run.status = "interrupted"
        run.finished_at = datetime.now(timezone.utc)
        run.last_message = "Спрян от потребителя"
        db.commit()

        if run.pid:
            _kill_process(run.pid)

    return RedirectResponse(url="/scrape/", status_code=303)


def _kill_process(pid: int) -> None:
    """Force-kill a process by PID. Uses taskkill on Windows (reliable with CREATE_NEW_PROCESS_GROUP)."""
    try:
        if sys.platform == "win32":
            subprocess.run(
                ["taskkill", "/F", "/PID", str(pid)],
                capture_output=True,
                timeout=5,
            )
        else:
            import os, signal
            os.kill(pid, signal.SIGTERM)
    except Exception:
        pass


@router.get("/progress/{run_id}")
async def progress_sse(run_id: str) -> StreamingResponse:
    """DB-polling SSE — survives uvicorn restarts."""

    async def event_stream():
        run_uuid = uuid.UUID(run_id)
        stale_threshold = timedelta(minutes=5)

        while True:
            try:
                with db_session() as s:
                    row = s.execute(
                        text("SELECT * FROM scrape_runs WHERE id = :id"),
                        {"id": run_uuid},
                    ).mappings().first()
            except Exception:
                yield _sse("error", {"message": "DB error"})
                break

            if row is None:
                yield _sse("error", {"message": "Run not found"})
                break

            data = dict(row)
            status = data.get("status", "")

            # Dead-process detection
            heartbeat = data.get("last_heartbeat_at")
            if status == "running" and heartbeat:
                if datetime.now(timezone.utc) - heartbeat > stale_threshold:
                    try:
                        with db_session() as s:
                            s.execute(text(
                                "UPDATE scrape_runs SET status='interrupted', finished_at=now() "
                                "WHERE id=:id AND status='running'"
                            ), {"id": run_uuid})
                    except Exception:
                        pass
                    data["status"] = "interrupted"
                    status = "interrupted"

            yield _sse("progress", data)

            if status in ("completed", "failed", "interrupted"):
                yield _sse("done", data)
                break

            await asyncio.sleep(1)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/history", response_class=HTMLResponse)
async def scrape_history(request: Request, db: Session = Depends(get_db)):
    recent_runs = (
        db.query(ScrapeRun)
        .order_by(ScrapeRun.started_at.desc())
        .limit(50)
        .all()
    )
    return templates.TemplateResponse(
        request, "scrape_history.html", {"runs": recent_runs}
    )


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"
