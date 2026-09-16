"""One way to reach the platform from a tool: through the gateway, as the caller.

🚨 THIS FILE IS THE SECURITY BOUNDARY FOR EVERY TOOL THAT TOUCHES CUSTOMER DATA.

A tool calling a backend with a service token would reach every bot in every
workspace, and the MCP grant map would be bypassed the moment a request
originated inside this service instead of at the gateway. The grants would still
be *configured*, and would protect nothing.

So: every call goes to the GATEWAY (never a service host), re-presenting the
credential the client sent, and the gateway makes exactly the decision it would
have made for a direct call. One enforcement point, no second opinion here.

It lives in its own module so a second tool file cannot quietly reach for httpx
and skip all of that.
"""

from __future__ import annotations

import os
from typing import Any

import httpx
import structlog

from src.protocol.models import TenantContext
from src.tools._caller import caller_auth_headers, has_caller_credential

logger = structlog.get_logger(__name__)

#: 🚨 The gateway, not a service host. Going straight to core-api or cms-core
#: would skip the grants plugin, which is the only thing deciding whether this
#: credential may touch this route at all.
_GATEWAY_URL = os.getenv("GATEWAY_URL", "https://localhost:8000").rstrip("/")
_TIMEOUT = 30.0
_VERIFY = os.getenv("MCP_UPSTREAM_VERIFY_TLS", "").strip().lower() in ("1", "true", "yes")

#: 🚨 Live updates ON for every write a tool makes. The gateway already stamps
#: `X-Shielva-Via: mcp`, so core-api would publish anyway; sending it explicitly
#: means these tools stream even if that header is ever dropped, and states the
#: intent at the call site. A browser's own saves stay silent — it sends neither.
LIVE = {"sse": "true"}


class ToolCallError(RuntimeError):
    """An upstream refusal, surfaced to the model as text it can act on."""


def fail(message: str, **fields: Any) -> dict[str, Any]:
    logger.warning("tool_call_failed", error=message, **fields)
    return {"status": "failed", "error": message}


async def call(
    method: str,
    path: str,
    tenant_context: TenantContext,
    *,
    params: dict[str, str] | None = None,
    json: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """One request to the gateway, as the caller.

    🚨 Refuses rather than falling back to an unauthenticated call. Without the
    caller's credential the gateway would either reject us or — worse, if this
    service ever gains one of its own — answer as the service, quietly stepping
    outside the grant map.
    """
    if not has_caller_credential():
        raise ToolCallError(
            "No caller credential on this request, so the workspace cannot be reached "
            "on your behalf. Reconnect the Shielva connector and try again."
        )

    headers = caller_auth_headers()
    headers["X-Tenant-ID"] = tenant_context.tenant_id
    headers["Content-Type"] = "application/json"

    async with httpx.AsyncClient(verify=_VERIFY, timeout=_TIMEOUT) as client:
        resp = await client.request(
            method,
            f"{_GATEWAY_URL}{path}",
            headers=headers,
            params=params,
            json=json,
        )

    if resp.status_code in (401, 403):
        # 🚨 Say WHICH thing was refused. "403" alone sends a model into a retry
        # loop; naming the grant tells the person reading the transcript what to
        # change in the admin screen.
        raise ToolCallError(
            f"Your MCP key is not permitted to {method} {path}. A platform owner controls this under MCP API Access."
        )
    if resp.status_code == 404:
        raise ToolCallError(f"Not found: {path}")
    if resp.status_code >= 400:
        raise ToolCallError(f"{method} {path} failed with {resp.status_code}: {resp.text[:200]}")

    try:
        return resp.json()
    except ValueError:
        return {}
