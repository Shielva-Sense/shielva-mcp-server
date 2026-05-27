"""Value objects for the LLM context.

These mirror the OpenAI chat-completions shape (message role/content,
tool_calls, finish_reason) because every modern provider speaks that
dialect via LiteLLM. We name them in our own vocabulary so adapter
substitution doesn't leak vendor types into the domain.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, NewType, Optional, Tuple


ModelId = NewType("ModelId", str)


class MessageRole(str, Enum):
    SYSTEM    = "system"
    USER      = "user"
    ASSISTANT = "assistant"
    TOOL      = "tool"


class FinishReason(str, Enum):
    """Why the provider stopped generating.

    ``STOP``      — natural end of message.
    ``LENGTH``    — hit max_tokens.
    ``TOOL_CALLS`` — model emitted tool calls (caller must execute + re-prompt).
    ``CONTENT_FILTER`` — provider's safety system intervened.
    ``OTHER``     — adapter-defined catch-all.
    """
    STOP            = "stop"
    LENGTH          = "length"
    TOOL_CALLS      = "tool_calls"
    CONTENT_FILTER  = "content_filter"
    OTHER           = "other"


@dataclass(frozen=True, slots=True)
class LLMToolCall:
    """One tool call emitted by the model.

    ``arguments`` is the raw JSON string from the provider (we don't
    json.loads here because invalid JSON should bubble up to the
    application layer, which decides whether to surface the parse
    error or retry the LLM call).
    """
    id:        str
    name:      str
    arguments: str  # JSON string

    @property
    def args_or_empty(self) -> Dict[str, Any]:
        """Best-effort parse; ``{}`` on any error. Useful when the
        caller wants to ignore malformed args (the LLM is unlikely
        to produce them, but the type system says it might)."""
        import json
        try:
            v = json.loads(self.arguments or "{}")
            return v if isinstance(v, dict) else {}
        except (ValueError, TypeError):
            return {}


@dataclass(frozen=True, slots=True)
class LLMMessage:
    """One turn in the conversation.

    Constraints (enforced by the application layer, not the VO):
        * ``role=tool``       implies ``tool_call_id`` is non-empty.
        * ``role=assistant``  may carry ``tool_calls`` instead of
                              ``content``.
        * ``role=system/user`` carry only ``content``.
    """
    role:         MessageRole
    content:      str                       = ""
    tool_calls:   Tuple[LLMToolCall, ...]  = ()
    tool_call_id: str                       = ""
    name:         str                       = ""  # tool name on role=tool


@dataclass(frozen=True, slots=True)
class LLMUsage:
    prompt_tokens:     int = 0
    completion_tokens: int = 0
    total_tokens:      int = 0


@dataclass(frozen=True, slots=True)
class LLMRequest:
    """One call to the provider.

    ``tools`` is the OpenAI tool-schema array (list of
    ``{"type":"function","function":{...}}``). Pass-through verbatim
    because LiteLLM expects exactly that shape; provider adapters
    translate if their wire format differs.
    """
    messages:    Tuple[LLMMessage, ...]
    model:       Optional[ModelId] = None    # None ⇒ provider default
    max_tokens:  int               = 1024
    temperature: float             = 0.1
    tools:       Optional[Tuple[Dict[str, Any], ...]] = None
    tool_choice: Optional[Any]     = None     # "auto" / "none" / {function:{name}}
    stream:      bool              = False


@dataclass(frozen=True, slots=True)
class LLMResponse:
    """What the provider returned for one ``complete`` call.

    Exactly one of ``content`` or ``tool_calls`` is meaningful per
    spec — when ``finish_reason == TOOL_CALLS`` the caller MUST
    execute the calls and re-prompt; when it's ``STOP`` the
    ``content`` is the final assistant message.
    """
    content:       str                      = ""
    tool_calls:    Tuple[LLMToolCall, ...]  = ()
    model:         ModelId                  = ModelId("")
    finish_reason: FinishReason             = FinishReason.OTHER
    usage:         LLMUsage                 = field(default_factory=LLMUsage)
