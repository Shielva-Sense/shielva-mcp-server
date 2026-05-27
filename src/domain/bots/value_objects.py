"""Value objects for the bots context."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, NewType, Tuple


BotId   = NewType("BotId",   str)
BotName = NewType("BotName", str)


class BotStatus(str, Enum):
    DRAFT     = "draft"
    TESTING   = "testing"
    ACTIVE    = "active"
    SUSPENDED = "suspended"


@dataclass(frozen=True, slots=True)
class PromptVariable:
    """One ``{{var}}`` placeholder found in a prompt template."""
    name:     str
    required: bool = False


# Compile once — used by PromptTemplate.parse_variables.
_VARIABLE_PATTERN = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")


@dataclass(frozen=True, slots=True)
class PromptTemplate:
    """A bot's system prompt template.

    Why a value object (and not just ``str``):
        * ``render(arguments)`` knows the substitution rules.
        * ``variables`` is derivable but cacheable.
        * The MCP spec's ``prompts/get`` returns rendered messages —
          having the rendering logic on the VO keeps that contract
          honoured uniformly regardless of which adapter sourced
          the template (Mongo bot config, in-memory fallback, …).
    """
    text: str

    @property
    def variables(self) -> Tuple[PromptVariable, ...]:
        seen: set[str] = set()
        out: list[PromptVariable] = []
        for match in _VARIABLE_PATTERN.finditer(self.text):
            name = match.group(1)
            if name in seen:
                continue
            seen.add(name)
            # We don't infer required vs optional — the spec lets
            # callers pass any subset; missing variables remain
            # `{{name}}` in the rendered output, which is the
            # client's signal that it was not supplied.
            out.append(PromptVariable(name=name, required=False))
        return tuple(out)

    def render(self, arguments: Dict[str, str]) -> str:
        if not arguments:
            return self.text
        out = self.text
        for k, v in arguments.items():
            out = out.replace("{{" + str(k) + "}}", str(v))
        return out
