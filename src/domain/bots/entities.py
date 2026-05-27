"""Bot aggregate root."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

from .value_objects import BotId, BotName, BotStatus, PromptTemplate


@dataclass(slots=True)
class Bot:
    """A tenant-scoped bot configuration.

    ``model_settings`` is intentionally a flat dict so adapters can
    surface provider-specific knobs (temperature, top_p, etc.)
    without forcing a domain change. The HandleQuery use case
    (slice 4b) reads it and constructs an :class:`LLMRequest`.
    """
    id:             BotId
    tenant_id:      str
    name:           BotName
    description:    str
    status:         BotStatus
    prompt_template: PromptTemplate
    kb_ids:         Tuple[str, ...]    = field(default_factory=tuple)
    tool_whitelist: Tuple[str, ...]    = field(default_factory=tuple)
    model_settings: Dict[str, str]     = field(default_factory=dict)

    def is_visible_to_tenant(self, tenant_id: str) -> bool:
        return self.tenant_id == tenant_id
