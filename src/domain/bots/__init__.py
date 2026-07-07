"""Bots bounded context.

A *bot* is a tenant-scoped configuration object that binds:
    * A KB set (via ``kb_ids``)
    * A tool whitelist
    * A system prompt template
    * A model config (provider + temperature defaults)

Bots are NOT MCP-protocol entities — the spec has no ``bots/*``
methods. They're shielva-mcp's internal grouping. But the spec's
``prompts/*`` surface needs a way to enumerate prompt templates,
and bots are how we group those today (one prompt per bot). The
dispatcher exposes them as ``bot/<bot_id>``.

Public surface:
    Entities      : Bot
    Value objects : BotId, BotName, BotStatus, PromptTemplate
    Ports         : BotRepository
    Errors        : BotNotFoundError
"""

from .entities import Bot
from .errors import BotNotFoundError
from .repositories import BotRepository
from .value_objects import (
    BotId,
    BotName,
    BotStatus,
    PromptTemplate,
    PromptVariable,
)

__all__ = [
    "Bot",
    "BotId",
    "BotName",
    "BotNotFoundError",
    "BotRepository",
    "BotStatus",
    "PromptTemplate",
    "PromptVariable",
]
