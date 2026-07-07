"""Bots-context use cases.

* get_bot       — single bot lookup with tenant scoping.
* list_bots     — tenant-wide listing (backs MCP ``prompts/list``).
* render_prompt — given (bot_id, arguments), produce the rendered
                  prompt text (backs MCP ``prompts/get``).

ProvisionBot + ActivateBot use cases land in slice 4b alongside the
Mongo-backed BotRepository adapter.
"""

from .services import BotApplicationService

__all__ = ["BotApplicationService"]
