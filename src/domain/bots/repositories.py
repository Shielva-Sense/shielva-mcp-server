"""Bots-context port."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..shared.tenant import TenantContext
from .entities import Bot
from .value_objects import BotId


class BotRepository(ABC):
    @abstractmethod
    async def get(self, bot_id: BotId, *, tenant: TenantContext) -> Bot | None:
        """Returns None for unknown id OR foreign tenant. The
        adapter SHOULD log the foreign-tenant case for ops visibility
        (it indicates a misconfigured client or an attack)."""

    @abstractmethod
    async def list_for_tenant(self, *, tenant: TenantContext) -> list[Bot]:
        """Every bot owned by this tenant. Empty when none."""

    @abstractmethod
    async def save(self, bot: Bot) -> None:
        """Upsert. Used by ProvisionBot use case (slice 4b)."""
