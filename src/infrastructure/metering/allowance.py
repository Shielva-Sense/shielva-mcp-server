"""Whether a workspace still has the tokens it is about to spend.

🚨 FAILS OPEN, always. This sits in front of every completion, so a slow or
unreachable billing service must never stop a customer's bot answering. The
cost of letting a few calls through past the line is a small overage; the cost
of refusing them because a settings lookup timed out is an outage in somebody's
phone line.

🚨 Cached, and deliberately not for long. The balance moves with every call, so
a long cache lets a workspace run far past its allowance; a very short one puts
an HTTP round trip inside the answer path. Thirty seconds is the compromise —
worst case a workspace overruns by half a minute of traffic.
"""

from __future__ import annotations

import time
from typing import Any

import httpx
import structlog

from config.settings import get_settings

logger = structlog.get_logger(__name__)

_TTL_SECONDS = 30.0
# tenant_id -> (checked_at, exhausted)
_CACHE: dict[str, tuple[float, bool]] = {}


def _configured() -> tuple[str, str] | None:
    try:
        settings = get_settings()
    except Exception:
        return None
    url = (settings.usage_ingest_url or "").strip()
    secret = settings.usage_ingest_secret.get_secret_value() if settings.usage_ingest_secret else ""
    return (url.rstrip("/"), secret) if url and secret else None


def cached_exhausted(tenant_id: str) -> bool:
    row = _CACHE.get(tenant_id)
    if not row:
        return False
    checked_at, exhausted = row
    return exhausted if time.monotonic() - checked_at <= _TTL_SECONDS else False


async def is_exhausted(tenant_id: str | None) -> bool:
    """True only when we KNOW the allowance is spent.

    Anything else — no tenant, no config, a timeout, a bad response — is False.
    """
    if not tenant_id:
        return False

    row = _CACHE.get(tenant_id)
    if row and time.monotonic() - row[0] <= _TTL_SECONDS:
        return row[1]

    cfg = _configured()
    if cfg is None:
        return False
    base, secret = cfg

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(2.0)) as client:
            resp = await client.get(
                f"{base}/api/v1/internal/usage/allowance",
                params={"tenant_id": tenant_id},
                headers={"X-Internal-Secret": secret},
            )
    except httpx.HTTPError as exc:
        logger.warning("mcp.allowance_check_failed", error=str(exc)[:200])
        return False

    if resp.status_code != 200:
        return False
    try:
        body: dict[str, Any] = resp.json()
    except ValueError:
        return False

    # Either ceiling being hit stops the spend: the token count, or the output
    # half of it, which is what actually costs money.
    exhausted = bool(body.get("exhausted")) or bool(body.get("output_exhausted"))
    _CACHE[tenant_id] = (time.monotonic(), exhausted)
    return exhausted
