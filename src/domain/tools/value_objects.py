"""Value objects for the tools context.

All immutable + equal-by-value. No I/O.

Notes
-----
:class:`ToolSchema` is a thin wrapper over a JSON Schema dict. We do
NOT model JSON Schema as an algebraic structure — the spec hands the
schema verbatim to the LLM, which expects the standard ``type``,
``properties``, ``required`` keys. Validating it here would be
duplicative; the catalogue adapter validates at registration time.

:class:`ToolResult` mirrors the MCP spec's ``CallToolResult`` shape:
a list of content blocks (text|image), plus an ``is_error`` flag.
Domain code never constructs the protocol-wire JSON — the interface
layer translates this VO into the spec's JSON envelope.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, NewType, Tuple, Union


# A tool name is a stable string identifier. NewType keeps it nominal
# so accidental ``str -> ToolName`` is a type error in mypy.
ToolName = NewType("ToolName", str)


@dataclass(frozen=True, slots=True)
class ToolSchema:
    """JSON Schema for the tool's input.

    Carried as a plain dict because the LLM gets the raw schema. We
    keep it framework-free — no Pydantic model — so the domain stays
    importable from anywhere.
    """
    json_schema: Dict[str, Any]


# ── Content blocks (MCP spec) ───────────────────────────────────────

@dataclass(frozen=True, slots=True)
class ToolText:
    text: str
    type: str = "text"


@dataclass(frozen=True, slots=True)
class ToolImage:
    """Base64-encoded image; MCP spec requires the mime type so the
    client knows how to render it."""
    data:     str   # base64
    mimeType: str
    type:     str = "image"


# Union type for tool content. Spec adds ``audio`` and ``resource``
# in 2025-03-26 — model them in this union when the time comes.
ToolContentBlock = Union[ToolText, ToolImage]


@dataclass(frozen=True, slots=True)
class ToolResult:
    """A tool invocation outcome — what the LLM gets back.

    ``is_error=True`` is NOT a JSON-RPC error: spec says tool errors
    surface as a normal result with the error flag so the LLM can
    decide whether to retry. The interface adapter translates this
    accordingly.
    """
    content:  Tuple[ToolContentBlock, ...]
    is_error: bool = False

    @classmethod
    def text(cls, text: str, *, is_error: bool = False) -> "ToolResult":
        """Convenience: single-block text result."""
        return cls(content=(ToolText(text=text),), is_error=is_error)

    @classmethod
    def failure(cls, message: str) -> "ToolResult":
        return cls.text(message, is_error=True)
