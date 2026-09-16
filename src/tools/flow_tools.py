"""Dialog-flow tools — what lets Claude actually build a bot's conversation.

Registered from ``src/main.py``:

    from src.tools.flow_tools import register_flow_tools
    register_flow_tools(tool_registry)

Tools:
  shielva_list_bots       — the workspace's bots, so a flow can be addressed
  shielva_get_flow        — read a bot's current graph
  shielva_add_flow_node   — append ONE node (and optionally one edge), save
  shielva_save_flow       — replace the whole graph in one write

WHY ``add_flow_node`` EXISTS ALONGSIDE ``save_flow``

``POST /bots/{id}/flow`` replaces ``nodes[]`` and ``edges[]`` wholesale — there
is no per-node endpoint. A model building a ten-node flow would therefore make
one write and the designer would jump from empty to finished.

🚨 The owner asked to WATCH nodes appear. That is only possible if each node is
its own write, so ``add_flow_node`` does read-modify-write: fetch the graph,
append, save. Ten nodes, ten saves, ten SSE events, ten visible steps.

🚨 THAT MAKES IT LOST-UPDATE PRONE, and it is handled rather than ignored: the
node is appended to the graph as READ BACK a moment earlier, so a person
dragging nodes in the browser at the same time can have their edit overwritten.
``expected_node_count`` lets the caller refuse to write onto a graph that
changed underneath it; a model looping over its own nodes passes the count it
last saw and finds out instead of silently clobbering.

🚨 EVERY CALL GOES THROUGH THE GATEWAY AS THE CALLER (see ``_caller.py``). A
service token here would reach every bot in every workspace and would bypass the
MCP grant map entirely — the grants would still be configured and would protect
nothing.
"""

from __future__ import annotations

import os
from typing import Any

import httpx
import structlog

from src.protocol.models import TenantContext, ToolDefinition
from src.tools._caller import caller_auth_headers, has_caller_credential

logger = structlog.get_logger(__name__)

#: 🚨 The gateway, not core-api directly. Going straight to the service would
#: skip the grants plugin, which is the only thing deciding whether this
#: credential may touch this route at all.
_GATEWAY_URL = os.getenv("GATEWAY_URL", "https://localhost:8000").rstrip("/")
_TIMEOUT = 30.0
_VERIFY = os.getenv("MCP_UPSTREAM_VERIFY_TLS", "").strip().lower() in ("1", "true", "yes")

#: 🚨 Live updates ON for everything here. The gateway already stamps
#: `X-Shielva-Via: mcp` so core-api would publish anyway; sending it explicitly
#: means these tools stream even if that header is ever dropped, and documents
#: the intent at the call site. A browser's own saves stay silent — it sends
#: neither the header nor this flag.
_LIVE = {"sse": "true"}


class FlowToolError(RuntimeError):
    """An upstream refusal, surfaced to the model as text it can act on."""


def _fail(message: str, **fields: Any) -> dict[str, Any]:
    logger.warning("flow_tool_failed", error=message, **fields)
    return {"status": "failed", "error": message}


async def _call(
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
        raise FlowToolError(
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
        raise FlowToolError(
            f"Your MCP key is not permitted to {method} {path}. A platform owner controls this under MCP API Access."
        )
    if resp.status_code == 404:
        raise FlowToolError(f"Not found: {path}")
    if resp.status_code >= 400:
        raise FlowToolError(f"{method} {path} failed with {resp.status_code}: {resp.text[:200]}")

    try:
        return resp.json()
    except ValueError:
        return {}


# ── handlers ──────────────────────────────────────────────────────────


async def shielva_list_bots(tenant_context: TenantContext) -> dict[str, Any]:
    """List the bots in this workspace, with their ids."""
    try:
        data = await _call("GET", "/bots/", tenant_context)
    except FlowToolError as exc:
        return _fail(str(exc))

    raw = data.get("bots") if isinstance(data, dict) else data
    bots = [
        {"id": b.get("id"), "name": b.get("name") or b.get("bot_name") or "", "status": b.get("status", "")}
        for b in (raw or [])
        if isinstance(b, dict)
    ]
    return {"status": "ok", "count": len(bots), "bots": bots}


async def shielva_get_flow(tenant_context: TenantContext, bot_id: str) -> dict[str, Any]:
    """Read a bot's dialog flow: its nodes and the edges between them."""
    if not bot_id:
        return _fail("bot_id is required")
    try:
        data = await _call("GET", f"/bots/{bot_id}/flow", tenant_context)
    except FlowToolError as exc:
        return _fail(str(exc), bot_id=bot_id)

    flow = data.get("flow", data) if isinstance(data, dict) else {}
    nodes = flow.get("nodes") or []
    edges = flow.get("edges") or []
    return {"status": "ok", "bot_id": bot_id, "node_count": len(nodes), "nodes": nodes, "edges": edges}


async def shielva_save_flow(
    tenant_context: TenantContext,
    bot_id: str,
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Replace a bot's whole flow. Prefer add_flow_node when building up a flow.

    🚨 This REPLACES the graph. Anything already there and not in `nodes` is
    gone, including work somebody did in the browser a second ago.
    """
    if not bot_id:
        return _fail("bot_id is required")
    if not isinstance(nodes, list):
        return _fail("nodes must be a list")

    try:
        data = await _call(
            "POST",
            f"/bots/{bot_id}/flow",
            tenant_context,
            params=_LIVE,
            json={"nodes": nodes, "edges": edges or []},
        )
    except FlowToolError as exc:
        return _fail(str(exc), bot_id=bot_id)

    flow = data.get("flow", {}) if isinstance(data, dict) else {}
    return {
        "status": "saved",
        "bot_id": bot_id,
        "node_count": len(flow.get("nodes") or nodes),
        "live_update": True,
    }


async def shielva_add_flow_node(
    tenant_context: TenantContext,
    bot_id: str,
    node: dict[str, Any],
    edge_from: str | None = None,
    expected_node_count: int | None = None,
) -> dict[str, Any]:
    """Append ONE node to a bot's flow, optionally joined from an existing node.

    Build a flow by calling this once per node: each call is its own save, so
    somebody watching the designer sees the nodes appear one at a time.
    """
    if not bot_id:
        return _fail("bot_id is required")
    if not isinstance(node, dict) or not node.get("id"):
        return _fail("node must be an object with an 'id'")

    try:
        current = await _call("GET", f"/bots/{bot_id}/flow", tenant_context)
    except FlowToolError as exc:
        return _fail(str(exc), bot_id=bot_id)

    flow = current.get("flow", current) if isinstance(current, dict) else {}
    nodes = list(flow.get("nodes") or [])
    edges = list(flow.get("edges") or [])

    # 🚨 Optimistic concurrency. Read-modify-write against a graph a person may
    # be editing in the browser: without this the last writer silently wins and
    # their dragged node disappears with no error anywhere.
    if expected_node_count is not None and len(nodes) != expected_node_count:
        return _fail(
            f"The flow changed underneath this edit: expected {expected_node_count} "
            f"nodes, found {len(nodes)}. Re-read the flow before adding.",
            bot_id=bot_id,
        )

    if any(isinstance(n, dict) and n.get("id") == node["id"] for n in nodes):
        return _fail(f"A node with id {node['id']!r} already exists in this flow", bot_id=bot_id)

    nodes.append(node)
    if edge_from:
        if not any(isinstance(n, dict) and n.get("id") == edge_from for n in nodes):
            return _fail(f"edge_from names a node that is not in the flow: {edge_from!r}", bot_id=bot_id)
        edges.append({"id": f"{edge_from}->{node['id']}", "source": edge_from, "target": node["id"]})

    try:
        await _call(
            "POST",
            f"/bots/{bot_id}/flow",
            tenant_context,
            params=_LIVE,
            json={"nodes": nodes, "edges": edges},
        )
    except FlowToolError as exc:
        return _fail(str(exc), bot_id=bot_id)

    logger.info("flow_node_added", bot_id=bot_id, node_id=node["id"], total=len(nodes))
    return {
        "status": "added",
        "bot_id": bot_id,
        "node_id": node["id"],
        "node_count": len(nodes),
        "live_update": True,
    }


# ── definitions ───────────────────────────────────────────────────────

#: 🚨 TWO INDEPENDENT FLAGS, TWO DIFFERENT SURFACES. Getting these the wrong way
#: round is how the codegen / TMS tools ended up advertised to every customer.
#:
#:   requires_permissions=[]   — VISIBLE in MCP `tools/list`. Unlike codegen and
#:                               TMS these ARE product surface: a customer admin
#:                               is meant to see them. What they may actually
#:                               reach is decided by the gateway from their MCP
#:                               grant map on every call, so there is one
#:                               enforcement point rather than a second opinion
#:                               here.
#:
#:   enabled_by_default=False  — NOT in a bot's runtime toolset. This flag feeds
#:                               `get_tools_for_bot`, a completely separate
#:                               surface: it decides what a customer's live bot
#:                               can call while answering somebody. True here
#:                               would let a bot rewrite its own dialog flow
#:                               mid-conversation. It does NOT hide the tool from
#:                               `tools/list` — `Tool.is_permitted_for` never
#:                               reads it.
FLOW_TOOL_DEFINITIONS: list[tuple[ToolDefinition, Any]] = [
    (
        ToolDefinition(
            name="shielva_list_bots",
            description="List the bots in this Shielva workspace, with their ids and status.",
            parameters=[],
            requires_permissions=[],
            enabled_by_default=False,
        ),
        shielva_list_bots,
    ),
    (
        ToolDefinition(
            name="shielva_get_flow",
            description=(
                "Read a bot's dialog flow — its nodes and the edges between them. "
                "Call before editing so you know what is already there."
            ),
            parameters=[
                {"name": "bot_id", "type": "string", "description": "The bot's id", "required": True},
            ],
            requires_permissions=[],
            enabled_by_default=False,
        ),
        shielva_get_flow,
    ),
    (
        ToolDefinition(
            name="shielva_add_flow_node",
            description=(
                "Append one node to a bot's dialog flow, optionally joined from an existing "
                "node. Call once per node when building a flow: each call saves separately, "
                "so anyone watching the dialog designer sees the nodes appear one at a time. "
                "Pass expected_node_count (from the last read) to be told, rather than "
                "silently overwrite, if somebody edited the flow in the meantime."
            ),
            parameters=[
                {"name": "bot_id", "type": "string", "description": "The bot's id", "required": True},
                {
                    "name": "node",
                    "type": "object",
                    "description": "The node to add. Must carry a unique 'id'.",
                    "required": True,
                },
                {
                    "name": "edge_from",
                    "type": "string",
                    "description": "Id of an existing node to draw an edge from.",
                    "required": False,
                },
                {
                    "name": "expected_node_count",
                    "type": "integer",
                    "description": "How many nodes you last saw. Refuses the write if it changed.",
                    "required": False,
                },
            ],
            requires_permissions=[],
            enabled_by_default=False,
        ),
        shielva_add_flow_node,
    ),
    (
        ToolDefinition(
            name="shielva_save_flow",
            description=(
                "Replace a bot's entire dialog flow in one write. REPLACES everything — "
                "anything not in `nodes` is deleted, including edits somebody just made in "
                "the browser. Prefer shielva_add_flow_node when building a flow up."
            ),
            parameters=[
                {"name": "bot_id", "type": "string", "description": "The bot's id", "required": True},
                {"name": "nodes", "type": "array", "description": "The complete node list", "required": True},
                {"name": "edges", "type": "array", "description": "The complete edge list", "required": False},
            ],
            requires_permissions=[],
            enabled_by_default=False,
        ),
        shielva_save_flow,
    ),
]


def register_flow_tools(registry) -> None:
    """Register the dialog-flow tools into the MCP tool registry."""
    for definition, handler in FLOW_TOOL_DEFINITIONS:
        registry.register(definition, handler)
    logger.info("flow_tools_registered", count=len(FLOW_TOOL_DEFINITIONS))
