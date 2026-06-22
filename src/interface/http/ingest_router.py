"""
RAG Ingestion management endpoints.

POST /mcp/v1/ingest/start           — Start bulk ingestion job
GET  /mcp/v1/ingest                 — List recent jobs
GET  /mcp/v1/ingest/stats           — Vector-store statistics
GET  /mcp/v1/ingest/{job_id}        — Poll single job
POST /mcp/v1/ingest/{job_id}/cancel — Cancel running job
GET  /mcp/v1/ingest/schedule        — Get auto-schedule config
PUT  /mcp/v1/ingest/schedule        — Update auto-schedule config

Jobs are tracked in-process. The collector subprocess streams stdout which
we parse to extract per-ecosystem and total progress counters.

Auto-schedule: APScheduler (AsyncIOScheduler) runs a daily cron job at a
configurable UTC hour/minute. Config is persisted to schedule_config.json
next to the mcp-server root so it survives restarts.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Literal, Optional

import structlog
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE            = Path(__file__).resolve()
# _HERE = .../mcp-server/src/interface/http/ingest_router.py
# parents[0]=http  [1]=interface  [2]=src  [3]=mcp-server  [4]=shielva-mcp repo root.
# The collector lives at the repo root, NOT under mcp-server — earlier this used
# four .parents (→ mcp-server/security-fix-collector, which does not exist) and
# every ingest returned 503 "Collector script not found".
COLLECTOR_DIR    = _HERE.parents[4] / "security-fix-collector"
COLLECTOR_SCRIPT = COLLECTOR_DIR / "collector.py"
SCHEDULE_CONFIG_PATH = _HERE.parent.parent.parent / "schedule_config.json"
_COLLECTOR_ENV   = COLLECTOR_DIR / ".env"


def _read_collector_env_key(key: str) -> str:
    """
    Read a key directly from the collector's .env file, bypassing the OS
    environment. This prevents a stale shell-level env var from overriding
    the value in the collector's own .env.
    """
    try:
        for line in _COLLECTOR_ENV.read_text().splitlines():
            line = line.strip()
            if line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            if k.strip() == key:
                return v.strip()
    except Exception:
        pass
    return ""

ALL_ECOSYSTEMS = ["npm", "PyPI", "Go", "Maven", "RubyGems", "NuGet"]
DEFAULT_LIMIT  = 10_000

# ── Models ────────────────────────────────────────────────────────────────────

class IngestionJob(BaseModel):
    job_id:               str
    status:               Literal["pending", "running", "completed", "failed", "cancelled"]
    ecosystems:           List[str]
    limit:                int
    timeout_minutes:      Optional[int] = None
    started_at:           Optional[str] = None
    completed_at:         Optional[str] = None
    current_ecosystem:    Optional[str] = None
    advisories_processed: int = 0
    entries_stored:       int = 0
    error_message:        Optional[str] = None
    logs:                 List[str] = Field(default_factory=list)
    triggered_by:         Literal["manual", "schedule"] = "manual"


class StartIngestionRequest(BaseModel):
    ecosystems:           List[str]      = Field(default_factory=lambda: list(ALL_ECOSYSTEMS))
    limit:                int            = DEFAULT_LIMIT
    github_token:         Optional[str]  = None  # fine-grained PAT: repo/commit/advisory access
    github_search_token:  Optional[str]  = None  # classic PAT: /search/commits (fine-grained can't)
    timeout_minutes:      Optional[int]  = None  # None = no timeout


class StatsResponse(BaseModel):
    total_entries:       int
    table:               str = "security_fix_entries"
    collector_available: bool
    fetched_at:          str


class ScheduleConfig(BaseModel):
    """Auto-ingestion schedule config (persisted to schedule_config.json)."""
    enabled:         bool          = False
    hour:            int           = 0       # UTC 0-23 (default midnight)
    minute:          int           = 0       # 0-59
    ecosystems:      List[str]     = Field(default_factory=lambda: list(ALL_ECOSYSTEMS))
    limit:           int           = DEFAULT_LIMIT
    timeout_minutes: Optional[int] = 120     # None = no timeout; default 2 h
    last_run_at:     Optional[str] = None    # ISO timestamp of last successful run (persisted)
    next_run:        Optional[str] = None    # computed at read time, not stored


# ── In-memory state ───────────────────────────────────────────────────────────

_jobs:  Dict[str, IngestionJob]                 = {}
_procs: Dict[str, asyncio.subprocess.Process]   = {}

_stats_cache:    Optional[StatsResponse] = None
_stats_cache_ts: float                   = 0.0
_STATS_TTL = 60.0

# ── APScheduler ───────────────────────────────────────────────────────────────

_scheduler = AsyncIOScheduler(timezone="UTC")


def _load_schedule() -> ScheduleConfig:
    try:
        if SCHEDULE_CONFIG_PATH.exists():
            data = json.loads(SCHEDULE_CONFIG_PATH.read_text())
            data.pop("next_run", None)
            return ScheduleConfig(**data)
    except Exception as exc:
        logger.warning("schedule_config_load_failed", error=str(exc))
    return ScheduleConfig()


def _save_schedule(config: ScheduleConfig) -> None:
    """Persist schedule config to disk (excludes the computed next_run field)."""
    data = config.model_dump(exclude={"next_run"})
    SCHEDULE_CONFIG_PATH.write_text(json.dumps(data, indent=2))


def _record_last_run() -> None:
    """Stamp last_run_at in schedule_config.json after a successful ingestion."""
    try:
        cfg = _load_schedule()
        cfg.last_run_at = datetime.now(timezone.utc).isoformat()
        _save_schedule(cfg)
    except Exception as exc:
        logger.warning("record_last_run_failed", error=str(exc))


def _apply_schedule(config: ScheduleConfig) -> None:
    _scheduler.remove_all_jobs()
    if config.enabled:
        _scheduler.add_job(
            _scheduled_ingest_job,
            CronTrigger(hour=config.hour, minute=config.minute, timezone="UTC"),
            id="auto_ingest",
            replace_existing=True,
        )
        logger.info("auto_schedule_applied", hour=config.hour, minute=config.minute)
    else:
        logger.info("auto_schedule_disabled")


async def _scheduled_ingest_job() -> None:
    """Cron callback — fires at the configured UTC time."""
    cfg = _load_schedule()
    if not cfg.enabled:
        return
    for job in _jobs.values():
        if job.status in ("pending", "running"):
            logger.info("scheduled_ingest_skipped", reason="job_already_running", job_id=job.job_id)
            return
    if not COLLECTOR_SCRIPT.exists():
        logger.error("scheduled_ingest_skipped", reason="collector_not_found")
        return

    job_id = str(uuid.uuid4())
    job = IngestionJob(
        job_id=job_id,
        status="pending",
        ecosystems=cfg.ecosystems,
        limit=cfg.limit,
        timeout_minutes=cfg.timeout_minutes,
        started_at=datetime.now(timezone.utc).isoformat(),
        triggered_by="schedule",
    )
    _jobs[job_id] = job
    asyncio.create_task(
        _run_ingestion(job_id, None, cfg.ecosystems, cfg.limit, cfg.timeout_minutes)
    )
    logger.info("scheduled_ingest_triggered", job_id=job_id, ecosystems=cfg.ecosystems)


def start_scheduler() -> None:
    """Start APScheduler and apply persisted schedule config. Call from app lifespan.

    Catch-up logic: if schedule is enabled and the last successful run was
    more than 25 hours ago (i.e. a midnight run was missed while the server was
    down), we trigger an immediate run on startup so no cycle is ever skipped.
    """
    cfg = _load_schedule()
    _apply_schedule(cfg)
    if not _scheduler.running:
        _scheduler.start()
        logger.info("ingest_scheduler_started")

    if cfg.enabled:
        _maybe_catchup(cfg)


def _maybe_catchup(cfg: ScheduleConfig) -> None:
    """If the last scheduled run was missed while the server was down, run now."""
    if not cfg.last_run_at:
        return   # never run before — don't auto-trigger on first startup
    try:
        last = datetime.fromisoformat(cfg.last_run_at)
        hours_since = (datetime.now(timezone.utc) - last).total_seconds() / 3600
        if hours_since >= 25:
            logger.info("ingest_catchup_triggered", hours_since=round(hours_since, 1))
            asyncio.get_event_loop().call_soon(
                lambda: asyncio.ensure_future(_scheduled_ingest_job())
            )
    except Exception as exc:
        logger.warning("catchup_check_failed", error=str(exc))


def stop_scheduler() -> None:
    """Gracefully stop scheduler on app shutdown."""
    if _scheduler.running:
        _scheduler.shutdown(wait=False)
        logger.info("ingest_scheduler_stopped")


# ── Router ────────────────────────────────────────────────────────────────────

router = APIRouter(prefix="/mcp/v1/ingest", tags=["ingest"])


# ── Stats ─────────────────────────────────────────────────────────────────────

@router.get("/stats", response_model=StatsResponse)
async def get_stats() -> StatsResponse:
    """Return current vector-store statistics (60 s TTL cache)."""
    global _stats_cache, _stats_cache_ts

    now = time.monotonic()
    if _stats_cache and (now - _stats_cache_ts) < _STATS_TTL:
        return _stats_cache

    if not COLLECTOR_SCRIPT.exists():
        result = StatsResponse(
            total_entries=0,
            collector_available=False,
            fetched_at=datetime.now(timezone.utc).isoformat(),
        )
        _stats_cache = result
        _stats_cache_ts = now
        return result

    try:
        proc = await asyncio.create_subprocess_exec(
            "python3", str(COLLECTOR_SCRIPT), "--mode", "stats",
            cwd=str(COLLECTOR_DIR),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=20)
        text = stdout.decode("utf-8", errors="replace")
        m = re.search(r"Total entries:\s*([\d,]+)", text)
        count = int(m.group(1).replace(",", "")) if m else 0
        result = StatsResponse(
            total_entries=count,
            collector_available=True,
            fetched_at=datetime.now(timezone.utc).isoformat(),
        )
    except Exception as exc:
        logger.warning("ingest_stats_failed", error=str(exc))
        result = StatsResponse(
            total_entries=0,
            collector_available=True,
            fetched_at=datetime.now(timezone.utc).isoformat(),
        )

    _stats_cache = result
    _stats_cache_ts = now
    return result


def _invalidate_stats_cache() -> None:
    global _stats_cache_ts
    _stats_cache_ts = 0.0


# ── Schedule endpoints ────────────────────────────────────────────────────────

@router.get("/schedule", response_model=ScheduleConfig)
async def get_schedule() -> ScheduleConfig:
    """Return current auto-schedule configuration plus computed next_run."""
    cfg = _load_schedule()
    if cfg.enabled:
        apscheduler_job = _scheduler.get_job("auto_ingest")
        if apscheduler_job and apscheduler_job.next_run_time:
            cfg.next_run = apscheduler_job.next_run_time.isoformat()
    return cfg


@router.put("/schedule", response_model=ScheduleConfig)
async def update_schedule(body: ScheduleConfig) -> ScheduleConfig:
    """Update auto-schedule configuration and immediately apply to scheduler."""
    body.hour   = max(0, min(23, body.hour))
    body.minute = max(0, min(59, body.minute))
    if body.timeout_minutes is not None:
        body.timeout_minutes = max(10, min(1440, body.timeout_minutes))  # 10 min – 24 h
    _save_schedule(body)
    _apply_schedule(body)
    logger.info("schedule_updated", enabled=body.enabled, hour=body.hour,
                minute=body.minute, timeout_minutes=body.timeout_minutes)
    return await get_schedule()


# ── List jobs ─────────────────────────────────────────────────────────────────

@router.get("", response_model=List[IngestionJob])
async def list_jobs() -> List[IngestionJob]:
    """Return last 10 jobs (most-recent first), each without full log lines."""
    ordered = sorted(_jobs.values(), key=lambda j: j.started_at or "", reverse=True)[:10]
    return [j.model_copy(update={"logs": j.logs[-5:]}) for j in ordered]


# ── Start ingestion ───────────────────────────────────────────────────────────

@router.post("/start", response_model=IngestionJob, status_code=202)
async def start_ingestion(body: StartIngestionRequest) -> IngestionJob:
    """Kick off a bulk ingestion job. Returns immediately; poll /{job_id} for progress."""
    for job in _jobs.values():
        if job.status in ("pending", "running"):
            raise HTTPException(
                status_code=409,
                detail=f"Job {job.job_id} is already running. Cancel it first.",
            )

    if not COLLECTOR_SCRIPT.exists():
        raise HTTPException(
            status_code=503,
            detail=f"Collector script not found at {COLLECTOR_SCRIPT}",
        )

    ecosystems = [e for e in body.ecosystems if e in ALL_ECOSYSTEMS]
    if not ecosystems:
        raise HTTPException(status_code=422, detail="No valid ecosystems selected")

    job_id = str(uuid.uuid4())
    job = IngestionJob(
        job_id=job_id,
        status="pending",
        ecosystems=ecosystems,
        limit=max(1, min(body.limit, 100_000)),
        timeout_minutes=body.timeout_minutes,
        started_at=datetime.now(timezone.utc).isoformat(),
        triggered_by="manual",
    )
    _jobs[job_id] = job
    asyncio.create_task(
        _run_ingestion(job_id, body.github_token, ecosystems, body.limit,
                       body.timeout_minutes, body.github_search_token)
    )
    logger.info("ingest_job_started", job_id=job_id, ecosystems=ecosystems,
                limit=body.limit, timeout_minutes=body.timeout_minutes)
    return job


async def _run_ingestion(
    job_id: str,
    github_token: Optional[str],
    ecosystems: List[str],
    limit: int,
    timeout_minutes: Optional[int] = None,
    github_search_token: Optional[str] = None,
) -> None:
    job = _jobs[job_id]
    job.status = "running"

    # ── Timeout watchdog ──────────────────────────────────────────────────────
    watchdog_task: Optional[asyncio.Task] = None
    if timeout_minutes and timeout_minutes > 0:
        async def _watchdog() -> None:
            await asyncio.sleep(timeout_minutes * 60)
            current = _jobs.get(job_id)
            if current and current.status == "running":
                current.status = "cancelled"
                current.error_message = (
                    f"Auto-cancelled: exceeded {timeout_minutes} min time limit"
                )
                logger.warning("ingest_timeout", job_id=job_id, timeout_minutes=timeout_minutes)
                proc = _procs.get(job_id)
                if proc:
                    try:
                        proc.terminate()
                    except Exception:
                        pass
        watchdog_task = asyncio.create_task(_watchdog())

    env = os.environ.copy()
    if github_token:
        env["GITHUB_TOKEN"] = github_token
    if github_search_token:
        env["GITHUB_SEARCH_TOKEN"] = github_search_token
    env["TARGET_ECOSYSTEMS"] = json.dumps(ecosystems)
    # Always inject GEMINI_API_KEY directly from the collector's .env file so
    # a stale OS-level env var can never override it.
    gemini_key = _read_collector_env_key("GEMINI_API_KEY")
    if gemini_key:
        env["GEMINI_API_KEY"] = gemini_key

    try:
        proc = await asyncio.create_subprocess_exec(
            "python3", str(COLLECTOR_SCRIPT),
            "--mode", "bulk",
            "--ecosystems", ",".join(ecosystems),
            "--limit", str(limit),
            cwd=str(COLLECTOR_DIR),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=env,
        )
        _procs[job_id] = proc

        assert proc.stdout is not None
        async for raw in proc.stdout:
            if job.status == "cancelled":
                break
            line = raw.decode("utf-8", errors="replace").rstrip()
            if not line:
                continue

            job.logs = (job.logs + [line])[-200:]

            m = re.search(r"Processing ecosystem:\s*([\w.+\-]+)", line)
            if m:
                job.current_ecosystem = m.group(1)

            m = re.search(r"total:\s*(\d+)", line)
            if m:
                job.entries_stored = int(m.group(1))

            m = re.search(r"advisories processed:\s*(\d+)", line)
            if m:
                job.advisories_processed = int(m.group(1))

        rc = await proc.wait()
        _procs.pop(job_id, None)

        if job.status == "cancelled":
            pass
        elif rc == 0:
            job.status = "completed"
            _invalidate_stats_cache()
            _record_last_run()   # persist timestamp for catch-up logic on next startup
            logger.info("ingest_job_completed", job_id=job_id,
                        entries=job.entries_stored, processed=job.advisories_processed)
        else:
            job.status = "failed"
            job.error_message = f"Collector exited with code {rc}"
            logger.error("ingest_job_failed", job_id=job_id, rc=rc)

    except Exception as exc:
        job.status = "failed"
        job.error_message = str(exc)
        _procs.pop(job_id, None)
        logger.error("ingest_job_exception", job_id=job_id, error=str(exc))
    finally:
        if watchdog_task and not watchdog_task.done():
            watchdog_task.cancel()
        job.completed_at = datetime.now(timezone.utc).isoformat()


# ── Get single job ────────────────────────────────────────────────────────────

@router.get("/{job_id}", response_model=IngestionJob)
async def get_job(job_id: str) -> IngestionJob:
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return _jobs[job_id]


# ── Cancel job ────────────────────────────────────────────────────────────────

@router.post("/{job_id}/cancel", response_model=IngestionJob)
async def cancel_job(job_id: str) -> IngestionJob:
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = _jobs[job_id]
    if job.status not in ("pending", "running"):
        raise HTTPException(
            status_code=409,
            detail=f"Cannot cancel a job in '{job.status}' state",
        )

    job.status = "cancelled"
    job.completed_at = datetime.now(timezone.utc).isoformat()

    proc = _procs.pop(job_id, None)
    if proc:
        try:
            proc.terminate()
            await asyncio.wait_for(proc.wait(), timeout=5)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    logger.info("ingest_job_cancelled", job_id=job_id)
    return job
