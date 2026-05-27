"""Bot application service — backs MCP prompts/* surface."""
from __future__ import annotations

from typing import Dict, List

import structlog

from src.domain.bots.entities import Bot
from src.domain.bots.errors import BotNotFoundError
from src.domain.bots.repositories import BotRepository
from src.domain.bots.value_objects import BotId
from src.domain.shared.tenant import TenantContext

logger = structlog.get_logger(__name__)


class BotApplicationService:
    def __init__(self, *, repository: BotRepository) -> None:
        self._repo = repository

    async def get_bot(self, *, bot_id: str, tenant: TenantContext) -> Bot:
        bot = await self._repo.get(BotId(bot_id), tenant=tenant)
        if bot is None or not bot.is_visible_to_tenant(tenant.tenant_id):
            raise BotNotFoundError(f"Bot not found: {bot_id}")
        return bot

    async def list_bots(self, *, tenant: TenantContext) -> List[Bot]:
        return await self._repo.list_for_tenant(tenant=tenant)

    async def render_prompt(
        self,
        *,
        bot_id:    str,
        tenant:    TenantContext,
        arguments: Dict[str, str],
    ) -> str:
        """Materialise the bot's prompt template with the supplied
        arguments. Unknown variables remain ``{{name}}`` in the
        output so the MCP client can see what wasn't supplied."""
        bot = await self.get_bot(bot_id=bot_id, tenant=tenant)
        return bot.prompt_template.render(arguments or {})
