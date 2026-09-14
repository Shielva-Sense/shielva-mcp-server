"""A bot's connector tools — the ones somebody turned on for it.

🚨 OPT-IN, and that is the whole design.

A workspace with seven connectors installed has roughly forty callable methods.
Putting all of them in front of the model on every turn would add forty tool
schemas to a prompt on a path where turn latency is the product — this codebase
has already spent three builds chasing voice latency — and would hand the model
forty ways to do something the bot was never meant to do. A bot that answers
questions about opening hours has no business holding `create_contact`.

So a connector tool reaches a bot only when it is switched on for that bot,
through the same per-bot mechanism the built-in tools already use. Nothing
changes for any bot until somebody enables something.

The enablement question is asked of the SAME registry that answers it for
built-ins, rather than of a second list, because two places to say "this bot may
use that tool" is how a bot ends up able to do something its configuration says
it cannot.
"""

from __future__ import annotations

from typing import Any

import structlog

from src.domain.shared.tenant import TenantContext

logger = structlog.get_logger(__name__)


async def specs_for_bot(
    *,
    catalogue: Any,
    registry: Any,
    bot_id: str,
    tenant: TenantContext,
    overrides: dict[str, bool] | None = None,
) -> list[Any]:
    """Enabled connector tools for this bot, as legacy `ToolSpec`s.

    Returns `[]` — never raises — when there is no catalogue, nothing enabled,
    or the connector runtime cannot be reached. A bot losing its connector tools
    for a turn is a degraded answer; a bot that cannot answer at all because the
    connector runtime blinked is an outage, and the second is much worse.
    """
    if catalogue is None or not bot_id:
        return []

    try:
        tools = await catalogue.list_for(tenant)
    except Exception as exc:
        logger.warning(
            "mcp.bot_connector_tools_unavailable",
            bot_id=bot_id,
            tenant_id=tenant.tenant_id,
            error=str(exc)[:200],
        )
        return []
    if not tools:
        return []

    from src.routing.llm_router import ToolSpec

    specs: list[Any] = []
    for tool in tools:
        name = str(tool.name)
        # 🚨 Default OFF. `_is_tool_enabled` falls back to a registered tool's
        # own default and returns False for a name it has never seen — and a
        # connector tool is never registered, so the fallback is exactly the
        # "off unless somebody said otherwise" this needs. An explicit override
        # or per-bot config still wins, which is how it gets turned on.
        if not registry._is_tool_enabled(name, bot_id, overrides):
            continue

        specs.append(
            ToolSpec(
                name=name,
                description=tool.description,
                parameters=tool.input_schema.json_schema,
                handler=_handler_for(catalogue=catalogue, tool=tool),
            )
        )

    if specs:
        logger.info(
            "mcp.bot_connector_tools",
            bot_id=bot_id,
            tenant_id=tenant.tenant_id,
            count=len(specs),
        )
    return specs


def _handler_for(*, catalogue: Any, tool: Any):
    """Bind one tool to the catalogue that owns it.

    🚨 The tenant comes from the CALL, not from this closure. Binding the tenant
    here would capture whichever tenant happened to be listing when the spec was
    built — and these specs are built per request precisely so that cannot
    happen, but a closure over the wrong one would reintroduce it silently.
    """

    async def _run(tenant_context: Any, **arguments: Any) -> Any:
        result = await catalogue.execute(
            tool=tool,
            arguments=arguments,
            tenant=_to_domain_tenant(tenant_context),
        )
        text = "\n".join(block.text for block in result.content if getattr(block, "text", None))
        if result.is_error:
            # Surfaced as a value, not an exception: the MCP spec's own
            # reasoning is that a tool error is something the model should see
            # and can retry or work around, rather than a crash in the loop.
            return {"error": text}
        return text

    return _run


def _to_domain_tenant(tenant_context: Any) -> TenantContext:
    """Legacy tenant object → the domain one the catalogue expects."""
    if isinstance(tenant_context, TenantContext):
        return tenant_context
    return TenantContext(
        tenant_id=getattr(tenant_context, "tenant_id", "") or "",
        user_id=getattr(tenant_context, "user_id", "") or "",
        user_email=getattr(tenant_context, "user_email", "") or "",
    )
