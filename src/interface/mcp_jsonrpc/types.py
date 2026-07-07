"""Pydantic models for JSON-RPC 2.0 + MCP spec message shapes.

Reference: https://spec.modelcontextprotocol.io/ (2024-11-05).

We use ``model_config = ConfigDict(extra="allow")`` on the envelope
classes so future spec extensions and unknown ``_meta`` fields don't
fail validation — the spec explicitly allows additional fields.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# ── JSON-RPC 2.0 envelopes ─────────────────────────────────────────────


class JsonRpcRequest(BaseModel):
    """Inbound request from client."""

    model_config = ConfigDict(extra="allow")

    jsonrpc: Literal["2.0"] = "2.0"
    id: str | int | None = None  # None ⇒ notification, no response
    method: str
    params: dict[str, Any] | None = None


class JsonRpcError(BaseModel):
    code: int
    message: str
    data: Any | None = None


class JsonRpcResponse(BaseModel):
    """Outbound response or error."""

    jsonrpc: Literal["2.0"] = "2.0"
    id: str | int | None = None
    result: Any | None = None
    error: JsonRpcError | None = None


# ── MCP common types ──────────────────────────────────────────────────


class Implementation(BaseModel):
    """Name + version of either client or server."""

    name: str
    version: str


class ServerCapabilities(BaseModel):
    """What the server supports. Each capability is an object so future
    fields can be added without breaking existing clients."""

    model_config = ConfigDict(extra="allow")

    tools: dict[str, Any] | None = None  # {"listChanged": bool}
    resources: dict[str, Any] | None = None  # {"subscribe": bool, "listChanged": bool}
    prompts: dict[str, Any] | None = None  # {"listChanged": bool}
    logging: dict[str, Any] | None = None  # {}
    experimental: dict[str, Any] | None = None


class ClientCapabilities(BaseModel):
    model_config = ConfigDict(extra="allow")

    roots: dict[str, Any] | None = None
    sampling: dict[str, Any] | None = None
    experimental: dict[str, Any] | None = None


# ── initialize ────────────────────────────────────────────────────────


class InitializeParams(BaseModel):
    protocolVersion: str
    capabilities: ClientCapabilities = Field(default_factory=ClientCapabilities)
    clientInfo: Implementation


class InitializeResult(BaseModel):
    protocolVersion: str
    capabilities: ServerCapabilities
    serverInfo: Implementation
    instructions: str | None = None


# ── tools/list, tools/call ────────────────────────────────────────────


class ToolListItem(BaseModel):
    """One tool in tools/list response. ``inputSchema`` MUST be a valid
    JSON Schema object (the spec is strict about this — clients pass it
    straight to the LLM as a function definition)."""

    name: str
    description: str = ""
    inputSchema: dict[str, Any]  # JSON Schema


class ToolListResult(BaseModel):
    tools: list[ToolListItem]
    nextCursor: str | None = None  # pagination — unused in v1


class ToolCallParams(BaseModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    _meta: dict[str, Any] | None = None


class ToolContentText(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ToolContentImage(BaseModel):
    type: Literal["image"] = "image"
    data: str  # base64
    mimeType: str


class ToolCallResult(BaseModel):
    """Spec result: a list of content blocks + isError flag."""

    content: list[ToolContentText | ToolContentImage]
    isError: bool = False


# ── resources/list, resources/read ────────────────────────────────────


class Resource(BaseModel):
    """One resource in resources/list. ``uri`` is the addressing key
    used by resources/read. ``mimeType`` lets clients pick a renderer."""

    uri: str
    name: str
    description: str | None = None
    mimeType: str | None = None


class ResourceListResult(BaseModel):
    resources: list[Resource]
    nextCursor: str | None = None


class ResourceContents(BaseModel):
    uri: str
    mimeType: str | None = None
    text: str | None = None  # for text resources
    blob: str | None = None  # for binary resources (base64)


class ResourceReadResult(BaseModel):
    contents: list[ResourceContents]


# ── prompts/list, prompts/get ─────────────────────────────────────────


class PromptArgument(BaseModel):
    name: str
    description: str | None = None
    required: bool = False


class Prompt(BaseModel):
    name: str
    description: str | None = None
    arguments: list[PromptArgument] = Field(default_factory=list)


class PromptListResult(BaseModel):
    prompts: list[Prompt]
    nextCursor: str | None = None


class PromptMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: ToolContentText  # we only emit text in v1


class PromptGetResult(BaseModel):
    description: str | None = None
    messages: list[PromptMessage]


# ── logging/setLevel ──────────────────────────────────────────────────


class LoggingSetLevelParams(BaseModel):
    level: Literal["debug", "info", "notice", "warning", "error", "critical", "alert", "emergency"]
