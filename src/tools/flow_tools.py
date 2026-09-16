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

from typing import Any

import structlog

from src.protocol.models import TenantContext, ToolDefinition
from src.tools._gateway import LIVE as _LIVE
from src.tools._gateway import ToolCallError as FlowToolError
from src.tools._gateway import call as _call
from src.tools._gateway import fail as _fail

logger = structlog.get_logger(__name__)


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
    edge_handle: str | None = None,
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
        # 🚨 A branch needs its OUTPUT named. A classify node routes each case down
        # the edge whose sourceHandle is that case's id; a condition by its branch
        # id; an ask by "fallback". Without the handle every edge here was an
        # unlabelled "next", so a flow built node-by-node could draw a straight
        # line and could not branch at all.
        new_edge = {
            "id": f"{edge_from}-{edge_handle}->{node['id']}" if edge_handle else f"{edge_from}->{node['id']}",
            "source": edge_from,
            "target": node["id"],
        }
        if edge_handle:
            new_edge["sourceHandle"] = edge_handle
        edges.append(new_edge)

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
                {
                    "name": "edge_handle",
                    "type": "string",
                    "description": 'Which OUTPUT of edge_from to join from: a classify case id, a condition branch id, or "fallback" on an ask. Omit for a plain next edge.',
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
