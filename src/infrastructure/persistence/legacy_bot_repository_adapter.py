"""Adapter wrapping the existing BotRegistry into the new
:class:`domain.bots.BotRepository` port.

The legacy registry talks to MongoDB ``customerService`` collection
and falls back to a mock when Mongo is absent. This adapter
preserves both modes — we just translate the dict-shaped bot
record into the new :class:`Bot` aggregate and apply tenant scoping
defensively.

A native Mongo adapter (without going through the legacy registry)
lands in slice 4b — at that point this wrapper goes away.
"""

from __future__ import annotations

from typing import Any

import structlog

from src.domain.bots.entities import Bot
from src.domain.bots.repositories import BotRepository
from src.domain.bots.value_objects import (
    BotId,
    BotName,
    BotStatus,
    PromptTemplate,
)
from src.domain.shared.tenant import TenantContext

logger = structlog.get_logger(__name__)


class LegacyBotRepositoryAdapter(BotRepository):
    def __init__(self, legacy_registry: Any) -> None:
        self._reg = legacy_registry

    async def get(self, bot_id: BotId, *, tenant: TenantContext) -> Bot | None:
        try:
            raw = await self._reg.get_bot(str(bot_id), tenant.tenant_id)
        except Exception as e:
            logger.warning(
                "bot_registry.get_bot failed",
                bot_id=str(bot_id),
                tenant_id=tenant.tenant_id,
                error=str(e),
            )
            return None
        if not raw:
            return None
        # Defence-in-depth: the legacy fallback returns a mock for
        # ANY bot_id when Mongo is unreachable. The mock has no
        # tenant binding, so we use the caller's tenant — same
        # behaviour as before this adapter existed.
        return _to_domain(raw, tenant_id=tenant.tenant_id)

    async def list_for_tenant(self, *, tenant: TenantContext) -> list[Bot]:
        # The legacy registry has no tenant-wide listing — it's
        # bot-by-id. Mongo path could enumerate the customer's
        # ``bots`` array, but the registry doesn't expose that
        # today. Return empty rather than fabricate.
        list_fn = getattr(self._reg, "list_for_tenant", None)
        if list_fn is None:
            return []
        try:
            rows = await list_fn(tenant.tenant_id)
        except Exception:
            return []
        return [_to_domain(r, tenant_id=tenant.tenant_id) for r in (rows or [])]

    async def save(self, bot: Bot) -> None:
        # ProvisionBot use case (slice 4b) will land here. For now
        # the legacy registry doesn't expose an upsert.
        logger.info(
            "mcp.bot_save_unsupported",
            bot_id=str(bot.id),
            note="Legacy BotRegistry has no upsert path",
        )


# ── helpers ────────────────────────────────────────────────────────


def _to_domain(raw: dict[str, Any], *, tenant_id: str) -> Bot:
    """Translate the legacy bot dict into a :class:`Bot`.

    Defensive on missing fields — the legacy mock + the Mongo
    document have slightly different shapes (the Mongo path adds
    ``kb_ids`` from ``kbs``; the mock has neither). We accept both.
    """
    prompt_text = ""
    prompt_config = raw.get("prompt_config") or {}
    if isinstance(prompt_config, dict):
        prompt_text = str(prompt_config.get("system_prompt") or prompt_config.get("custom_prompt") or "")
    if not prompt_text:
        prompt_text = str(raw.get("custom_prompt") or "")

    kb_ids: tuple[str, ...] = ()
    raw_kbs = raw.get("kb_ids") or raw.get("kbs") or []
    if isinstance(raw_kbs, list):
        kb_ids = tuple(str(k.get("id") if isinstance(k, dict) else k) for k in raw_kbs if k)

    model_settings: dict[str, str] = {}
    model_cfg = raw.get("model_config") or {}
    if isinstance(model_cfg, dict):
        for k, v in model_cfg.items():
            model_settings[str(k)] = str(v)

    return Bot(
        id=BotId(str(raw.get("id") or raw.get("bot_id") or "")),
        tenant_id=tenant_id,
        name=BotName(str(raw.get("name") or "")),
        description=str(raw.get("description") or ""),
        status=_parse_status(raw.get("status")),
        prompt_template=PromptTemplate(text=prompt_text),
        kb_ids=kb_ids,
        tool_whitelist=tuple(str(t) for t in (raw.get("tools") or ())),
        model_settings=model_settings,
    )


def _parse_status(value: Any) -> BotStatus:
    if value is None:
        return BotStatus.DRAFT
    try:
        return BotStatus(str(value))
    except ValueError:
        return BotStatus.DRAFT
