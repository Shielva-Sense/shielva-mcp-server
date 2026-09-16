"""Bot configuration tools — identity, intents, decision rules, signals, settings.

Registered from ``src/main.py`` alongside the flow tools.

WHERE EACH THING ACTUALLY LIVES, because it is not one service:

  identity        cms-core   /cms/api/v1/identity        the bot's persona
  intents         cms-core   /cms/api/v1/intents         what it recognises
  decision rules  cms-core   /cms/api/v1/signals/rules   when it acts
  action schemas  cms-core   /cms/api/v1/action-schemas  what it can do
  fact groups     cms-core   /cms/api/v1/fact-groups     what it remembers
  prompt, variables, capabilities, memory policy, signals, tools
                  core-api   /bots/{id}/…                per-bot settings

🚨 Both sit behind routes the MCP grant map already covers (`cms` and `bots`),
and every call goes through the gateway as the caller — see ``_gateway.py``. A
tool here never decides what the caller may reach; the gateway does, on every
call, from their own grant map.

🚨 IDENTITY HERE MEANS THE BOT'S PERSONA, not a person's account. The two are
unrelated and the name collision is a real trap: ``/cms/api/v1/identity`` is who
the BOT is — its tone, its name, how it introduces itself — while
``/identity/api/v1/…`` is the identity SERVICE, which manages users and is not
reachable from a customer's MCP key at all.
"""

from __future__ import annotations

from typing import Any

import structlog

from src.protocol.models import TenantContext, ToolDefinition
from src.tools._gateway import LIVE as _LIVE
from src.tools._gateway import ToolCallError
from src.tools._gateway import call as _call
from src.tools._gateway import fail as _fail

logger = structlog.get_logger(__name__)

_CMS = "/cms/api/v1"


def _items(data: Any, key: str = "items") -> list[dict[str, Any]]:
    """Upstreams differ: some return a bare list, some wrap it. Accept both
    rather than making the model guess which endpoint does which."""
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        for k in (key, "results", "data"):
            if isinstance(data.get(k), list):
                return [x for x in data[k] if isinstance(x, dict)]
    return []


# ── identity: who the bot is ──────────────────────────────────────────


async def shielva_list_identities(tenant_context: TenantContext, bot_id: str | None = None) -> dict[str, Any]:
    """List the bot personas (identities) in this workspace."""
    try:
        data = await _call("GET", f"{_CMS}/identity", tenant_context, params={"bot_id": bot_id} if bot_id else None)
    except ToolCallError as exc:
        return _fail(str(exc))
    rows = _items(data)
    return {"status": "ok", "count": len(rows), "identities": rows}


async def shielva_create_identity(
    tenant_context: TenantContext,
    name: str,
    bot_id: str,
    persona: str = "",
    tone: str = "",
    greeting: str = "",
) -> dict[str, Any]:
    """Create a bot persona: its name, how it speaks, how it opens a conversation."""
    if not name or not bot_id:
        return _fail("name and bot_id are required")
    body = {"name": name, "bot_id": bot_id, "persona": persona, "tone": tone, "greeting": greeting}
    try:
        data = await _call("POST", f"{_CMS}/identity", tenant_context, params=_LIVE, json=body)
    except ToolCallError as exc:
        return _fail(str(exc), bot_id=bot_id)
    return {"status": "created", "identity": data}


async def shielva_update_identity(
    tenant_context: TenantContext, identity_id: str, changes: dict[str, Any]
) -> dict[str, Any]:
    """Change fields on an existing persona. Only the keys you pass are touched."""
    if not identity_id or not isinstance(changes, dict) or not changes:
        return _fail("identity_id and a non-empty changes object are required")
    try:
        data = await _call("PUT", f"{_CMS}/identity/{identity_id}", tenant_context, params=_LIVE, json=changes)
    except ToolCallError as exc:
        return _fail(str(exc), identity_id=identity_id)
    return {"status": "updated", "identity": data}


async def shielva_activate_identity(tenant_context: TenantContext, identity_id: str) -> dict[str, Any]:
    """Make this persona the live one for its bot.

    🚨 Changes what callers hear on the next conversation. Activating is a
    separate act from editing on purpose — a draft persona can be written and
    reviewed without anybody being answered in its voice.
    """
    if not identity_id:
        return _fail("identity_id is required")
    try:
        await _call(
            "PATCH",
            f"{_CMS}/identity/{identity_id}/activate",
            tenant_context,
            params={"tenant_id": tenant_context.tenant_id, **_LIVE},
        )
    except ToolCallError as exc:
        return _fail(str(exc), identity_id=identity_id)
    return {"status": "activated", "identity_id": identity_id}


# ── intents: what the bot recognises ──────────────────────────────────


async def shielva_list_intents(tenant_context: TenantContext, bot_id: str | None = None) -> dict[str, Any]:
    """List intents — the things the bot knows how to recognise."""
    try:
        data = await _call("GET", f"{_CMS}/intents", tenant_context, params={"bot_id": bot_id} if bot_id else None)
    except ToolCallError as exc:
        return _fail(str(exc))
    rows = _items(data)
    return {"status": "ok", "count": len(rows), "intents": rows}


async def shielva_create_intent(
    tenant_context: TenantContext,
    name: str,
    bot_id: str,
    examples: list[str] | None = None,
    description: str = "",
) -> dict[str, Any]:
    """Create an intent with example phrasings the bot should match on."""
    if not name or not bot_id:
        return _fail("name and bot_id are required")
    body = {"name": name, "bot_id": bot_id, "examples": examples or [], "description": description}
    try:
        data = await _call("POST", f"{_CMS}/intents", tenant_context, params=_LIVE, json=body)
    except ToolCallError as exc:
        return _fail(str(exc), bot_id=bot_id)
    return {"status": "created", "intent": data}


async def shielva_update_intent(
    tenant_context: TenantContext, intent_id: str, changes: dict[str, Any]
) -> dict[str, Any]:
    """Change an intent — rename it, or add example phrasings."""
    if not intent_id or not isinstance(changes, dict) or not changes:
        return _fail("intent_id and a non-empty changes object are required")
    try:
        data = await _call("PUT", f"{_CMS}/intents/{intent_id}", tenant_context, params=_LIVE, json=changes)
    except ToolCallError as exc:
        return _fail(str(exc), intent_id=intent_id)
    return {"status": "updated", "intent": data}


# ── decision rules: when the bot acts ─────────────────────────────────


async def shielva_list_decision_rules(tenant_context: TenantContext) -> dict[str, Any]:
    """List decision rules — the signal conditions that make the bot act."""
    try:
        data = await _call("GET", f"{_CMS}/signals/rules", tenant_context)
    except ToolCallError as exc:
        return _fail(str(exc))
    rows = _items(data, key="rules")
    return {"status": "ok", "count": len(rows), "rules": rows}


async def shielva_create_decision_rule(
    tenant_context: TenantContext,
    name: str,
    rule: dict[str, Any],
) -> dict[str, Any]:
    """Create a decision rule: a condition over signals and what it triggers."""
    if not name or not isinstance(rule, dict) or not rule:
        return _fail("name and a non-empty rule object are required")
    try:
        data = await _call("POST", f"{_CMS}/signals/rules", tenant_context, params=_LIVE, json={"name": name, **rule})
    except ToolCallError as exc:
        return _fail(str(exc))
    return {"status": "created", "rule": data}


async def shielva_update_decision_rule(
    tenant_context: TenantContext, rule_id: str, changes: dict[str, Any]
) -> dict[str, Any]:
    """Change a decision rule's condition or its effect."""
    if not rule_id or not isinstance(changes, dict) or not changes:
        return _fail("rule_id and a non-empty changes object are required")
    try:
        data = await _call("PUT", f"{_CMS}/signals/rules/{rule_id}", tenant_context, params=_LIVE, json=changes)
    except ToolCallError as exc:
        return _fail(str(exc), rule_id=rule_id)
    return {"status": "updated", "rule": data}


# ── per-bot settings ──────────────────────────────────────────────────


async def shielva_get_bot_config(tenant_context: TenantContext, bot_id: str) -> dict[str, Any]:
    """Read a bot's configuration: capabilities, signals, memory policy, variables.

    One call rather than four, because a model about to change one of these
    should see the others — a capability toggled without knowing the memory
    policy is a change made half-blind.
    """
    if not bot_id:
        return _fail("bot_id is required")

    out: dict[str, Any] = {"status": "ok", "bot_id": bot_id}
    # 🚨 Each part is fetched independently and a refusal on one does not lose
    # the rest: a grant map may permit capabilities and withhold signals, and
    # reporting nothing in that case would read as "this bot has no config".
    for key, path in (
        ("capabilities", f"/bots/{bot_id}/capabilities"),
        ("signals", f"/bots/{bot_id}/signals"),
        ("memory_policy", f"/bots/{bot_id}/memory/policy"),
        ("variables", f"/bots/{bot_id}/variables"),
    ):
        try:
            out[key] = await _call("GET", path, tenant_context)
        except ToolCallError as exc:
            out[key] = {"unavailable": str(exc)}
    return out


async def shielva_set_bot_prompt(tenant_context: TenantContext, bot_id: str, prompt: str) -> dict[str, Any]:
    """Replace a bot's system prompt — the standing instruction it answers under."""
    if not bot_id or not prompt:
        return _fail("bot_id and prompt are required")
    try:
        await _call("PUT", f"/bots/{bot_id}/update-prompt", tenant_context, params=_LIVE, json={"prompt": prompt})
    except ToolCallError as exc:
        return _fail(str(exc), bot_id=bot_id)
    return {"status": "updated", "bot_id": bot_id, "field": "prompt"}


async def shielva_set_bot_variables(
    tenant_context: TenantContext, bot_id: str, variables: dict[str, Any]
) -> dict[str, Any]:
    """Set the variables a bot's flow and prompt can reference."""
    if not bot_id or not isinstance(variables, dict):
        return _fail("bot_id and a variables object are required")
    try:
        await _call("POST", f"/bots/{bot_id}/variables", tenant_context, params=_LIVE, json={"variables": variables})
    except ToolCallError as exc:
        return _fail(str(exc), bot_id=bot_id)
    return {"status": "updated", "bot_id": bot_id, "field": "variables"}


async def shielva_set_bot_capability(
    tenant_context: TenantContext, bot_id: str, capability: str, enabled: bool
) -> dict[str, Any]:
    """Turn one of a bot's capabilities on or off."""
    if not bot_id or not capability:
        return _fail("bot_id and capability are required")
    try:
        await _call(
            "PUT",
            f"/bots/{bot_id}/capabilities/{capability}",
            tenant_context,
            params=_LIVE,
            json={"enabled": bool(enabled)},
        )
    except ToolCallError as exc:
        return _fail(str(exc), bot_id=bot_id)
    return {"status": "updated", "bot_id": bot_id, "capability": capability, "enabled": bool(enabled)}


async def shielva_set_bot_memory_policy(
    tenant_context: TenantContext, bot_id: str, policy: dict[str, Any]
) -> dict[str, Any]:
    """Set what a bot remembers between conversations."""
    if not bot_id or not isinstance(policy, dict) or not policy:
        return _fail("bot_id and a non-empty policy object are required")
    try:
        await _call("PUT", f"/bots/{bot_id}/memory/policy", tenant_context, params=_LIVE, json=policy)
    except ToolCallError as exc:
        return _fail(str(exc), bot_id=bot_id)
    return {"status": "updated", "bot_id": bot_id, "field": "memory_policy"}


async def shielva_set_bot_signals(
    tenant_context: TenantContext, bot_id: str, signals: list[dict[str, Any]]
) -> dict[str, Any]:
    """Set the signals a bot emits — what decision rules can then act on."""
    if not bot_id or not isinstance(signals, list):
        return _fail("bot_id and a signals list are required")
    try:
        await _call("POST", f"/bots/{bot_id}/signals", tenant_context, params=_LIVE, json={"signals": signals})
    except ToolCallError as exc:
        return _fail(str(exc), bot_id=bot_id)
    return {"status": "updated", "bot_id": bot_id, "count": len(signals)}


# ── action schemas: what a bot can DO ─────────────────────────────────


async def shielva_list_action_schemas(tenant_context: TenantContext) -> dict[str, Any]:
    """List action schemas — the connector calls an `action` flow node can point at."""
    try:
        data = await _call("GET", f"{_CMS}/action-schemas", tenant_context)
    except ToolCallError as exc:
        return _fail(str(exc))
    rows = _items(data)
    return {"status": "ok", "count": len(rows), "action_schemas": rows}


async def shielva_create_action_schema(
    tenant_context: TenantContext,
    name: str,
    connector_type: str,
    operation: str,
    inputs: dict[str, Any] | None = None,
    description: str = "",
) -> dict[str, Any]:
    """Create an action schema: a named connector call a flow can invoke.

    🚨 Create this BEFORE the `action` node that points at it. A node whose
    actionId names nothing saves cleanly and that branch does nothing at
    runtime — see shielva_flow_guide.
    """
    if not name or not connector_type or not operation:
        return _fail("name, connector_type and operation are required")
    body = {
        "name": name,
        "connector_type": connector_type,
        "operation": operation,
        "inputs": inputs or {},
        "description": description,
    }
    try:
        data = await _call("POST", f"{_CMS}/action-schemas", tenant_context, params=_LIVE, json=body)
    except ToolCallError as exc:
        return _fail(str(exc))
    return {"status": "created", "action_schema": data}


async def shielva_update_action_schema(
    tenant_context: TenantContext, schema_id: str, changes: dict[str, Any]
) -> dict[str, Any]:
    """Change an action schema's inputs, operation or name."""
    if not schema_id or not isinstance(changes, dict) or not changes:
        return _fail("schema_id and a non-empty changes object are required")
    try:
        data = await _call("PUT", f"{_CMS}/action-schemas/{schema_id}", tenant_context, params=_LIVE, json=changes)
    except ToolCallError as exc:
        return _fail(str(exc), schema_id=schema_id)
    return {"status": "updated", "action_schema": data}


async def shielva_test_action_schema(
    tenant_context: TenantContext, schema_id: str, inputs: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Run an action schema against its connector with test inputs.

    🚨 This REALLY CALLS the connector — it sends the mail, creates the lead. It
    is a test of the wiring, not a dry run, so use inputs you are willing to have
    land in the customer's third-party account.
    """
    if not schema_id:
        return _fail("schema_id is required")
    try:
        data = await _call(
            "POST",
            f"{_CMS}/action-schemas/test",
            tenant_context,
            json={"schema_id": schema_id, "inputs": inputs or {}},
        )
    except ToolCallError as exc:
        return _fail(str(exc), schema_id=schema_id)
    return {"status": "tested", "result": data}


# ── painters: how a bot RENDERS ───────────────────────────────────────


async def shielva_list_painters(tenant_context: TenantContext) -> dict[str, Any]:
    """List painters — the card/table templates a `painter` node renders with."""
    try:
        data = await _call("GET", f"{_CMS}/painters", tenant_context)
    except ToolCallError as exc:
        return _fail(str(exc))
    rows = _items(data)
    return {"status": "ok", "count": len(rows), "painters": rows}


async def shielva_create_painter(
    tenant_context: TenantContext,
    name: str,
    painter_type: str = "card",
    template: dict[str, Any] | None = None,
    custom: bool = False,
) -> dict[str, Any]:
    """Create a painter — a card or table template for rendering data.

    `custom=False` builds from the stock template for `painter_type`; `custom=True`
    takes the `template` you supply verbatim. A custom painter is yours to keep
    correct — nothing validates its fields against the data a flow will feed it.
    """
    if not name:
        return _fail("name is required")
    if custom and not template:
        return _fail("a custom painter needs a template")
    body = {
        "name": name,
        "painter_type": painter_type,
        "custom": bool(custom),
        "template": template or {},
    }
    try:
        data = await _call("POST", f"{_CMS}/painters", tenant_context, params=_LIVE, json=body)
    except ToolCallError as exc:
        return _fail(str(exc))
    return {"status": "created", "painter": data}


async def shielva_publish_painter(tenant_context: TenantContext, painter_id: str) -> dict[str, Any]:
    """Publish a painter so flows can render with it.

    🚨 Separate from creating it, deliberately: a painter can be written and
    reviewed before any conversation renders with it.
    """
    if not painter_id:
        return _fail("painter_id is required")
    try:
        await _call("POST", f"{_CMS}/painters/{painter_id}/publish", tenant_context, params=_LIVE)
    except ToolCallError as exc:
        return _fail(str(exc), painter_id=painter_id)
    return {"status": "published", "painter_id": painter_id}


# ── definitions ───────────────────────────────────────────────────────

_P_BOT = {"name": "bot_id", "type": "string", "description": "The bot's id", "required": True}


def _tool(name: str, description: str, params: list[dict[str, Any]], handler: Any):
    """🚨 Same two flags as flow_tools, same reasoning:
    requires_permissions=[] makes it VISIBLE in tools/list (these are product
    surface, and what a caller may reach is the gateway's decision from their
    grant map); enabled_by_default=False keeps it OUT of a live bot's runtime
    toolset, where it would let a bot reconfigure itself mid-conversation.
    """
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


BOT_CONFIG_TOOL_DEFINITIONS: list[tuple[ToolDefinition, Any]] = [
    _tool(
        "shielva_list_identities",
        "List bot personas (identities) — a bot's name, tone and greeting. This is who the BOT is, not a user account.",
        [{"name": "bot_id", "type": "string", "description": "Filter to one bot", "required": False}],
        shielva_list_identities,
    ),
    _tool(
        "shielva_create_identity",
        "Create a bot persona: its name, how it speaks, and how it opens a conversation.",
        [
            {"name": "name", "type": "string", "description": "Persona name", "required": True},
            _P_BOT,
            {"name": "persona", "type": "string", "description": "Who the bot is", "required": False},
            {"name": "tone", "type": "string", "description": "How it speaks", "required": False},
            {"name": "greeting", "type": "string", "description": "Opening line", "required": False},
        ],
        shielva_create_identity,
    ),
    _tool(
        "shielva_update_identity",
        "Change fields on an existing bot persona. Only the keys you pass are touched.",
        [
            {"name": "identity_id", "type": "string", "description": "Persona id", "required": True},
            {"name": "changes", "type": "object", "description": "Fields to change", "required": True},
        ],
        shielva_update_identity,
    ),
    _tool(
        "shielva_activate_identity",
        "Make a persona the live one for its bot. Changes what callers hear on the next conversation.",
        [{"name": "identity_id", "type": "string", "description": "Persona id", "required": True}],
        shielva_activate_identity,
    ),
    _tool(
        "shielva_list_intents",
        "List intents — the things a bot knows how to recognise.",
        [{"name": "bot_id", "type": "string", "description": "Filter to one bot", "required": False}],
        shielva_list_intents,
    ),
    _tool(
        "shielva_create_intent",
        "Create an intent with example phrasings the bot should match on.",
        [
            {"name": "name", "type": "string", "description": "Intent name", "required": True},
            _P_BOT,
            {"name": "examples", "type": "array", "description": "Example phrasings", "required": False},
            {"name": "description", "type": "string", "description": "What it means", "required": False},
        ],
        shielva_create_intent,
    ),
    _tool(
        "shielva_update_intent",
        "Change an intent — rename it, or add example phrasings.",
        [
            {"name": "intent_id", "type": "string", "description": "Intent id", "required": True},
            {"name": "changes", "type": "object", "description": "Fields to change", "required": True},
        ],
        shielva_update_intent,
    ),
    _tool(
        "shielva_list_decision_rules",
        "List decision rules — the signal conditions that make a bot act.",
        [],
        shielva_list_decision_rules,
    ),
    _tool(
        "shielva_create_decision_rule",
        "Create a decision rule: a condition over signals, and what it triggers.",
        [
            {"name": "name", "type": "string", "description": "Rule name", "required": True},
            {"name": "rule", "type": "object", "description": "Condition and effect", "required": True},
        ],
        shielva_create_decision_rule,
    ),
    _tool(
        "shielva_update_decision_rule",
        "Change a decision rule's condition or its effect.",
        [
            {"name": "rule_id", "type": "string", "description": "Rule id", "required": True},
            {"name": "changes", "type": "object", "description": "Fields to change", "required": True},
        ],
        shielva_update_decision_rule,
    ),
    _tool(
        "shielva_get_bot_config",
        "Read a bot's configuration in one call: capabilities, signals, memory policy and variables.",
        [_P_BOT],
        shielva_get_bot_config,
    ),
    _tool(
        "shielva_set_bot_prompt",
        "Replace a bot's system prompt — the standing instruction it answers under.",
        [_P_BOT, {"name": "prompt", "type": "string", "description": "The new prompt", "required": True}],
        shielva_set_bot_prompt,
    ),
    _tool(
        "shielva_set_bot_variables",
        "Set the variables a bot's flow and prompt can reference.",
        [
            _P_BOT,
            {"name": "variables", "type": "object", "description": "Variable map", "required": True},
        ],
        shielva_set_bot_variables,
    ),
    _tool(
        "shielva_set_bot_capability",
        "Turn one of a bot's capabilities on or off.",
        [
            _P_BOT,
            {"name": "capability", "type": "string", "description": "Capability key", "required": True},
            {"name": "enabled", "type": "boolean", "description": "On or off", "required": True},
        ],
        shielva_set_bot_capability,
    ),
    _tool(
        "shielva_set_bot_memory_policy",
        "Set what a bot remembers between conversations.",
        [_P_BOT, {"name": "policy", "type": "object", "description": "Memory policy", "required": True}],
        shielva_set_bot_memory_policy,
    ),
    _tool(
        "shielva_set_bot_signals",
        "Set the signals a bot emits — what decision rules can then act on.",
        [_P_BOT, {"name": "signals", "type": "array", "description": "Signal list", "required": True}],
        shielva_set_bot_signals,
    ),
    _tool(
        "shielva_list_action_schemas",
        "List action schemas — the connector calls an `action` flow node can point at.",
        [],
        shielva_list_action_schemas,
    ),
    _tool(
        "shielva_create_action_schema",
        "Create an action schema: a named connector call a flow can invoke. Create this BEFORE "
        "the action node that points at it — a node whose actionId names nothing saves cleanly "
        "and then does nothing at runtime.",
        [
            {"name": "name", "type": "string", "description": "Schema name", "required": True},
            {"name": "connector_type", "type": "string", "description": "Connector type", "required": True},
            {"name": "operation", "type": "string", "description": "Operation to call", "required": True},
            {"name": "inputs", "type": "object", "description": "Input mapping", "required": False},
            {"name": "description", "type": "string", "description": "What it does", "required": False},
        ],
        shielva_create_action_schema,
    ),
    _tool(
        "shielva_update_action_schema",
        "Change an action schema's inputs, operation or name.",
        [
            {"name": "schema_id", "type": "string", "description": "Schema id", "required": True},
            {"name": "changes", "type": "object", "description": "Fields to change", "required": True},
        ],
        shielva_update_action_schema,
    ),
    _tool(
        "shielva_test_action_schema",
        "Run an action schema against its connector. REALLY CALLS IT — sends the mail, creates the "
        "lead. Use inputs you are willing to have land in the customer's third-party account.",
        [
            {"name": "schema_id", "type": "string", "description": "Schema id", "required": True},
            {"name": "inputs", "type": "object", "description": "Test inputs", "required": False},
        ],
        shielva_test_action_schema,
    ),
    _tool(
        "shielva_list_painters",
        "List painters — the card/table templates a `painter` flow node renders with.",
        [],
        shielva_list_painters,
    ),
    _tool(
        "shielva_create_painter",
        "Create a painter: a card or table template. custom=False uses the stock template for the "
        "type; custom=True takes your template verbatim and nothing validates it against the data "
        "a flow will feed it.",
        [
            {"name": "name", "type": "string", "description": "Painter name", "required": True},
            {"name": "painter_type", "type": "string", "description": "card or table", "required": False},
            {"name": "template", "type": "object", "description": "Template, for custom", "required": False},
            {"name": "custom", "type": "boolean", "description": "Custom template", "required": False},
        ],
        shielva_create_painter,
    ),
    _tool(
        "shielva_publish_painter",
        "Publish a painter so flows can render with it. Separate from creating it, so a painter can "
        "be reviewed before any conversation renders with it.",
        [{"name": "painter_id", "type": "string", "description": "Painter id", "required": True}],
        shielva_publish_painter,
    ),
]


def register_bot_config_tools(registry) -> None:
    """Register the bot-configuration tools into the MCP tool registry."""
    for definition, handler in BOT_CONFIG_TOOL_DEFINITIONS:
        registry.register(definition, handler)
    logger.info("bot_config_tools_registered", count=len(BOT_CONFIG_TOOL_DEFINITIONS))
