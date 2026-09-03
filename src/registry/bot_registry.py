import copy
import time
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

#: Marks the tenant-owned asset row in ``customerService``. Must match
#: core-api's ``app.services.workspace_assets.WORKSPACE_SCOPE``.
WORKSPACE_SCOPE = "workspace"

from config.settings import get_settings


class BotRegistry:
    """
    Registry for bot configurations.
    Fetches actual bot data from MongoDB.
    """

    def __init__(self, mongodb_client=None):
        self.mongodb_client = mongodb_client
        self.settings = get_settings()
        # FIX #7: short TTL cache for resolved bot configs, keyed by
        # (tenant_id, bot_id). Bot config + KB-group membership change rarely,
        # but get_bot() runs on every turn (and again on the rag_query tool
        # path) — one Mongo find_one + KB-group walk per turn. The cache key
        # ALWAYS includes tenant_id so a bot config can never leak across
        # tenants. Mirrors the monotonic-time + dict TTL pattern in
        # tenant_llm_resolver. Stores a copy so callers mutating the returned
        # dict can't corrupt the cached entry.
        self._cache: dict[tuple[str, str], tuple[float, dict[str, Any]]] = {}
        self._cache_ttl = max(0, getattr(self.settings, "bot_cache_ttl_seconds", 45))

    async def _find_customer(self, tenant_id: str, bot_id: str | None = None) -> dict[str, Any] | None:
        """The document holding this tenant's bot, plus its workspace groups.

        🚨 Was ``find_one({"tenant_id": tenant_id})``. That is ambiguous — a
        tenant has one ``customerService`` row per member PLUS a workspace row
        — and core-api has since moved bots, knowledge and groups onto the
        workspace row, leaving members' rows with ``bots: []``. Mongo returns
        whichever row it finds first, so this could hand back a member row,
        ``get_bot`` would not find the bot in it, and every turn would silently
        fall back to ``_get_mock_bot``: the caller answers with mock config
        instead of its real prompt and knowledge, and nothing raises.

        Verified in production: for one tenant the member row already holds
        nbots=0 while the workspace row holds the two real bots. It had not
        fired yet only because that tenant had taken no calls.

        Resolution order, and why it is not simply "read the workspace row":
        a tenant whose assets have not been adopted yet still keeps them on a
        member row, and core-api creates the workspace row EMPTY on first read
        — so a missing or empty workspace row is not proof the bot is absent.
        Find the row that actually contains the bot, then take the groups from
        the workspace row, which is where they now live.
        """
        if not self.mongodb_client:
            return None
        coll = self.mongodb_client[self.settings.mongodb_db_name].customerService

        ws = await coll.find_one({"tenant_id": tenant_id, "scope": WORKSPACE_SCOPE})

        if bot_id is not None and not any(
            isinstance(b, dict) and b.get("id") == bot_id for b in (ws or {}).get("bots") or []
        ):
            legacy = await coll.find_one({"tenant_id": tenant_id, "bots.id": bot_id})
            if legacy is not None:
                if ws is None:
                    return legacy
                # Bots from the row that has them; groups from the workspace
                # row, so a half-migrated tenant resolves both correctly.
                merged = dict(legacy)
                for field in ("kb_groups", "bot_groups"):
                    if ws.get(field):
                        merged[field] = ws[field]
                return merged

        if ws is not None:
            return ws
        return await coll.find_one({"tenant_id": tenant_id})

    def invalidate(self, tenant_id: str, bot_id: str) -> None:
        """Drop the cached config for a (tenant, bot). Call from provisioning
        paths (bot edited, KBs (re)assigned) so the next get_bot() re-reads
        Mongo instead of serving stale config for up to the TTL window."""
        self._cache.pop((tenant_id, bot_id), None)

    async def get_bot(self, bot_id: str, tenant_id: str) -> dict[str, Any]:
        """
        Get bot configuration by ID and tenant from CustomerProfile.customerService.
        """
        # FIX #7: serve from the short TTL cache when fresh. This also dedupes
        # the second identical fetch on the rag_query tool path within a turn.
        cache_key = (tenant_id, bot_id)
        if self._cache_ttl > 0:
            entry = self._cache.get(cache_key)
            if entry is not None and (time.monotonic() - entry[0]) < self._cache_ttl:
                return copy.deepcopy(entry[1])

        if not self.mongodb_client:
            logger.warning("MongoDB client not initialized in BotRegistry")
            return self._get_mock_bot(bot_id)

        try:
            customer = await self._find_customer(tenant_id, bot_id)

            if not customer:
                logger.warning("Customer not found for tenant", tenant_id=tenant_id)
                return self._get_mock_bot(bot_id)

            # Find specific bot in bots array
            bots = customer.get("bots", [])
            bot = next((b for b in bots if b.get("id") == bot_id), None)

            if not bot:
                logger.warning("Bot not found in customer profile", bot_id=bot_id)
                return self._get_mock_bot(bot_id)

            # Ensure kb_ids is present and formatted correctly
            # ShielvaAPI might store them as 'kbs' or 'kb_ids'
            # In ShielvaAPI/api/bots.py it pushes to 'bots.$.kbs'

            # Extract this bot's DIRECT KB IDs (core-api pushes id strings to
            # bots.$.kbs; older docs may use dicts or a kb_ids field).
            kb_ids = []
            if "kbs" in bot and isinstance(bot["kbs"], list):
                for kb in bot["kbs"]:
                    if isinstance(kb, dict):
                        kb_ids.append(kb.get("id"))
                    else:
                        kb_ids.append(kb)
            elif "kb_ids" in bot:
                kb_ids = bot.get("kb_ids", [])

            # Expand to the EFFECTIVE KB set: direct KBs ∪ KBs from the bot's
            # assigned KB-groups ∪ KBs (direct + via KB-groups) bound to any
            # bot-group the bot belongs to. Mirrors core-api
            # knowledge_groups.resolve_effective_kb_ids (kept in sync; the two
            # live in separate services and cannot share code without a per-query
            # round-trip).
            kb_ids = self._resolve_effective_kb_ids(customer, bot, kb_ids)

            bot["kb_ids"] = kb_ids
            bot["kbs"] = kb_ids  # keep kbs aligned with the resolved set for retrieval

            logger.info(
                "Fetched bot config",
                bot_id=bot_id,
                name=bot.get("name"),
                kb_count=len(kb_ids),
            )

            # FIX #7: cache the successfully resolved config (never the mock
            # fallback — we don't want to pin a transient DB error for the TTL).
            if self._cache_ttl > 0:
                self._cache[cache_key] = (time.monotonic(), copy.deepcopy(bot))

            return bot

        except Exception as e:
            logger.error("Error fetching bot from MongoDB", error=str(e))
            return self._get_mock_bot(bot_id)

    @staticmethod
    def _resolve_effective_kb_ids(customer: dict[str, Any], bot: dict[str, Any], direct_kb_ids: list) -> list:
        """Expand a bot's direct KBs with KB-groups and bot-groups it belongs to.

        Effective = direct ∪ KBs of the bot's kb_group_ids ∪ (for each bot-group
        containing this bot) that group's direct kbs + its kb_groups' KBs.
        Order-preserving + de-duped. Pure function over the customer doc.
        """

        def _clean(ids) -> list:
            seen, out = set(), []
            for x in ids or []:
                if isinstance(x, str) and x.strip() and x not in seen:
                    seen.add(x)
                    out.append(x)
            return out

        kb_groups = {g.get("group_id"): g for g in (customer.get("kb_groups") or [])}
        bot_groups = customer.get("bot_groups") or []
        bot_id = bot.get("id")

        effective: list = []

        def _add(ids):
            for kid in _clean(ids):
                if kid not in effective:
                    effective.append(kid)

        def _expand(gid):
            g = kb_groups.get(gid)
            if g:
                _add(g.get("kb_ids"))

        _add(direct_kb_ids)
        for gid in _clean(bot.get("kb_group_ids")):
            _expand(gid)
        for bg in bot_groups:
            if bot_id in _clean(bg.get("bot_ids")):
                _add(bg.get("kbs"))
                for gid in _clean(bg.get("kb_group_ids")):
                    _expand(gid)
        return effective

    def _get_mock_bot(self, bot_id: str) -> dict[str, Any]:
        """Fallback mock bot"""
        return {
            "id": bot_id,
            "name": f"Bot {bot_id} (Fallback)",
            "description": "Auto-generated bot configuration (DB Error)",
            "prompt_config": {
                "system_prompt": "You are a helpful AI assistant. Answer questions clearly and accurately based on the provided context.",
                "tool_instructions": "Use available tools if the user question requires specialized actions.",
            },
            "kb_ids": [],
            "model_config": {"model": "gemini-1.5-pro", "temperature": 0.1},
        }
