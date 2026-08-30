"""Report metered consumption to the subscription service.

Fire-and-forget by construction. Metering must never be the reason a customer
does not get their answer: every failure path here is swallowed and logged, and
the report is scheduled as a background task so no completion ever waits on it.

That trade is deliberate. A lost usage row is a billing gap to reconcile; a
raised exception on the completion path is a broken product.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

import httpx
import structlog

from config.settings import get_settings

logger = structlog.get_logger(__name__)

# Strong refs to in-flight reports. Without this the event loop is free to
# garbage-collect a bare create_task mid-flight and the write silently vanishes.
_IN_FLIGHT: set[asyncio.Task[Any]] = set()


def _configured() -> bool:
    """Whether metering has somewhere to report to.

    Settings are read lazily, never at import. This module is imported by the
    LLM provider, so resolving config at module scope would make a
    configuration problem break the completion path itself — the exact failure
    metering is supposed to be incapable of causing.
    """
    try:
        settings = get_settings()
    except Exception:
        return False
    return bool(settings.usage_ingest_url and settings.usage_ingest_secret.get_secret_value())


async def _post(payload: dict[str, Any]) -> None:
    settings = get_settings()
    url = f"{settings.usage_ingest_url.rstrip('/')}/api/v1/internal/usage/record"
    try:
        async with httpx.AsyncClient(timeout=settings.usage_ingest_timeout) as client:
            resp = await client.post(
                url,
                json=payload,
                headers={"X-Internal-Secret": settings.usage_ingest_secret.get_secret_value()},
            )
        if resp.status_code >= 400:
            logger.warning(
                "mcp.usage_report_rejected",
                status_code=resp.status_code,
                detail=resp.text[:200],
            )
    except Exception as exc:
        logger.warning("mcp.usage_report_failed", error=str(exc)[:200])


def report_llm_usage(
    *,
    tenant_id: str | None,
    total_tokens: int,
    model_ref: str | None = None,
    provider: str | None = None,
    request_id: str | None = None,
) -> None:
    """Schedule a usage report. Returns immediately; never raises.

    A dedupe key is generated per call so an at-least-once delivery cannot
    double-meter — the ingest side absorbs the repeat.
    """
    if not tenant_id or total_tokens <= 0 or not _configured():
        return

    payload = {
        "events": [
            {
                "tenant_id": tenant_id,
                "kind": "llm",
                "quantity": int(total_tokens),
                "provider": provider,
                "model_ref": model_ref,
                "request_id": request_id,
                "dedupe_key": f"llm:{request_id or uuid.uuid4()}",
            }
        ]
    }

    # Check for the loop BEFORE building the coroutine. Creating it first and
    # then failing to schedule leaves an un-awaited coroutine behind.
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running loop — nothing to schedule onto. Not worth failing over.
        logger.debug("mcp.usage_report_no_loop")
        return

    task = loop.create_task(_post(payload))
    _IN_FLIGHT.add(task)
    task.add_done_callback(_IN_FLIGHT.discard)
