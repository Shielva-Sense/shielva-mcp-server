"""Bot lifecycle and knowledge tools — create, test, deploy, ground.

🚨 WHY THIS MODULE EXISTS

The flow and configuration tools could shape an existing bot but not bring one
into being, talk to it, ship it, or give it anything to answer from. Asking
Claude to "build a demo agent for a firm" stopped at step one: there was no way
to create the bot the other twenty-eight tools operate on, and no way to check
the result short of a person opening the dialog designer. These close that loop:

    create -> configure (bot_config_tools) -> draw (flow_tools)
           -> ground (knowledge) -> test (chat) -> deploy

Every call goes through ``_gateway.call`` as the caller, so the grant map decides
what each key may reach — exactly as for the other tool modules. The customer
bot-builder allow list already names ``/bots/create``, ``/bots/*/chat`` and
``/bots/*/deploy``; these tools add no reach a key did not already have.

Paths and bodies are read from the core-api handlers, not assumed:
``create_bot`` reads name/description/icon/status and returns ``bot_id``;
``create_knowledge_base`` takes name/description and returns ``kb_id``;
``ingest_url_to_kb`` reads url/crawl/max_pages; ``chat_with_bot`` reads
message/token/studio; ``set_bot_kb_groups`` takes ``kb_group_ids`` embedded.
"""

from __future__ import annotations

import uuid
from typing import Any

import structlog

from src.protocol.models import TenantContext, ToolDefinition
from src.tools._gateway import ToolCallError
from src.tools._gateway import call as _call
from src.tools._gateway import fail as _fail

logger = structlog.get_logger(__name__)


async def shielva_create_bot(
    tenant_context: TenantContext,
    name: str,
    description: str = "",
    icon: str = "",
) -> dict[str, Any]:
    """Create a bot in this workspace. Returns its id — every other tool needs it."""
    if not (name or "").strip():
        return _fail("name is required")
    body: dict[str, Any] = {"name": name.strip(), "description": description or "", "status": "draft"}
    if icon:
        body["icon"] = icon
    try:
        data = await _call("POST", "/bots/create", tenant_context, json=body)
    except ToolCallError as exc:
        return _fail(str(exc))
    bot_id = data.get("bot_id") if isinstance(data, dict) else None
    if not bot_id:
        return _fail("The platform accepted the request but returned no bot_id.", response=data)
    return {"status": "created", "bot_id": bot_id, "name": name.strip()}


async def shielva_chat_with_bot(
    tenant_context: TenantContext,
    bot_id: str,
    message: str,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Send one message to a bot and return its reply — the test surface.

    Use it after building a flow to check the bot actually does what the flow
    says, before deploying. Pass the same ``context`` across turns to continue
    a conversation.
    """
    if not bot_id or not (message or "").strip():
        return _fail("bot_id and message are required")
    # 🚨 /bots/{id}/chat refuses a turn with no session ``token`` (400 "message
    # (or cta_event) and token required") — this tool sent none, so it never
    # worked. ``studio: true`` is the designer's own test path: core-api opens
    # the session on the first turn, bound to the CALLER's tenant and this bot,
    # so a made-up token reaches nothing it could not already reach. The same
    # token carries the conversation, so it rides back in ``context``.
    ctx = dict(context or {})
    conversation_id = str(ctx.pop("conversation_id", "") or "") or f"mcp-test-{uuid.uuid4().hex}"
    body: dict[str, Any] = {"message": message, "token": conversation_id, "studio": True}
    if ctx:
        body["context"] = ctx
    try:
        data = await _call("POST", f"/bots/{bot_id}/chat", tenant_context, json=body)
    except ToolCallError as exc:
        return _fail(str(exc), bot_id=bot_id)
    return {
        "status": "ok",
        "bot_id": bot_id,
        "response": data,
        # Pass this back as ``context`` to continue the same conversation.
        "context": {"conversation_id": conversation_id},
    }


async def shielva_deploy_bot(tenant_context: TenantContext, bot_id: str) -> dict[str, Any]:
    """Deploy a bot to its live channels. Test with shielva_chat_with_bot first."""
    if not bot_id:
        return _fail("bot_id is required")
    try:
        data = await _call("POST", f"/bots/{bot_id}/deploy", tenant_context)
    except ToolCallError as exc:
        return _fail(str(exc), bot_id=bot_id)
    return {"status": "deploying", "bot_id": bot_id, "deployment": data}


async def shielva_create_knowledge_base(
    tenant_context: TenantContext, name: str, description: str = ""
) -> dict[str, Any]:
    """Create a knowledge base an `answer` node can ground its replies in."""
    if not (name or "").strip():
        return _fail("name is required")
    try:
        data = await _call(
            "POST", "/knowledge/create", tenant_context, json={"name": name.strip(), "description": description or ""}
        )
    except ToolCallError as exc:
        return _fail(str(exc))
    kb_id = (data.get("kb_id") or data.get("id")) if isinstance(data, dict) else None
    if not kb_id:
        return _fail("The platform accepted the request but returned no kb_id.", response=data)
    return {"status": "created", "kb_id": kb_id, "name": name.strip()}


async def shielva_ingest_url(
    tenant_context: TenantContext,
    kb_id: str,
    url: str,
    crawl: bool = False,
    max_pages: int = 20,
) -> dict[str, Any]:
    """Pull a web page (or, with crawl, a site) into a knowledge base."""
    if not kb_id or not (url or "").strip():
        return _fail("kb_id and url are required")
    body = {"url": url.strip(), "crawl": bool(crawl), "max_pages": max(1, int(max_pages or 1))}
    try:
        data = await _call("POST", f"/knowledge/{kb_id}/ingest-url", tenant_context, json=body)
    except ToolCallError as exc:
        return _fail(str(exc), kb_id=kb_id)
    return {"status": "ingesting", "kb_id": kb_id, "url": url.strip(), "response": data}


async def shielva_set_bot_knowledge_groups(
    tenant_context: TenantContext, bot_id: str, kb_group_ids: list[str]
) -> dict[str, Any]:
    """Link knowledge groups to a bot — what its `answer` nodes may draw on."""
    if not bot_id:
        return _fail("bot_id is required")
    ids = [str(i) for i in (kb_group_ids or []) if str(i).strip()]
    try:
        data = await _call("PUT", f"/bots/{bot_id}/kb-groups", tenant_context, json={"kb_group_ids": ids})
    except ToolCallError as exc:
        return _fail(str(exc), bot_id=bot_id)
    return {"status": "updated", "bot_id": bot_id, "kb_group_ids": ids, "response": data}


# ── definitions ───────────────────────────────────────────────────────

_P_BOT = {"name": "bot_id", "type": "string", "description": "The bot's id", "required": True}


def _tool(name: str, description: str, params: list[dict[str, Any]], handler: Any):
    """Same two flags as the other product tool modules: visible in tools/list
    (the gateway's grant map decides reach), and never in a live bot's runtime
    toolset (a bot must not create or deploy bots mid-conversation)."""
    return (
        ToolDefinition(
            name=name,
            description=description,
            parameters=params,
            requires_permissions=[],
            enabled_by_default=False,
        ),
        handler,
    )


LIFECYCLE_TOOL_DEFINITIONS: list[tuple[ToolDefinition, Any]] = [
    _tool(
        "shielva_create_bot",
        "Create a bot in this workspace and return its id. Do this first when building a new agent.",
        [
            {"name": "name", "type": "string", "description": "Bot name", "required": True},
            {"name": "description", "type": "string", "description": "What the bot is for", "required": False},
            {"name": "icon", "type": "string", "description": "Optional icon", "required": False},
        ],
        shielva_create_bot,
    ),
    _tool(
        "shielva_chat_with_bot",
        "Send a message to a bot and get its reply. Use this to TEST a flow before deploying.",
        [
            _P_BOT,
            {"name": "message", "type": "string", "description": "What the user says", "required": True},
            {
                "name": "context",
                "type": "object",
                "description": "Conversation context; reuse across turns",
                "required": False,
            },
        ],
        shielva_chat_with_bot,
    ),
    _tool(
        "shielva_deploy_bot",
        "Deploy a bot to its live channels. Publishes it — test with shielva_chat_with_bot first.",
        [_P_BOT],
        shielva_deploy_bot,
    ),
    _tool(
        "shielva_create_knowledge_base",
        "Create a knowledge base for a bot's `answer` nodes to ground replies in.",
        [
            {"name": "name", "type": "string", "description": "Knowledge base name", "required": True},
            {"name": "description", "type": "string", "description": "What it holds", "required": False},
        ],
        shielva_create_knowledge_base,
    ),
    _tool(
        "shielva_ingest_url",
        "Pull a web page, or with crawl a site, into a knowledge base.",
        [
            {"name": "kb_id", "type": "string", "description": "Knowledge base id", "required": True},
            {"name": "url", "type": "string", "description": "Page or site URL", "required": True},
            {"name": "crawl", "type": "boolean", "description": "Follow links on the site", "required": False},
            {"name": "max_pages", "type": "integer", "description": "Page cap when crawling", "required": False},
        ],
        shielva_ingest_url,
    ),
    _tool(
        "shielva_set_bot_knowledge_groups",
        "Link knowledge groups to a bot so its `answer` nodes can use them.",
        [
            _P_BOT,
            {
                "name": "kb_group_ids",
                "type": "array",
                "description": "Knowledge group ids",
                "required": True,
                "items": {"type": "string"},
            },
        ],
        shielva_set_bot_knowledge_groups,
    ),
]


def register_lifecycle_tools(registry: Any) -> None:
    for definition, handler in LIFECYCLE_TOOL_DEFINITIONS:
        registry.register(definition, handler)
    logger.info("lifecycle_tools_registered", count=len(LIFECYCLE_TOOL_DEFINITIONS))
