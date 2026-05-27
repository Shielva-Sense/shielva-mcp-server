"""Pydantic models for JSON-RPC 2.0 + MCP spec message shapes.

Reference: https://spec.modelcontextprotocol.io/ (2024-11-05).

We use ``model_config = ConfigDict(extra="allow")`` on the envelope
classes so future spec extensions and unknown ``_meta`` fields don't
fail validation — the spec explicitly allows additional fields.
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, ConfigDict, Field


# ── JSON-RPC 2.0 envelopes ─────────────────────────────────────────────

class JsonRpcRequest(BaseModel):
    """Inbound request from client."""
    model_config = ConfigDict(extra="allow")

    jsonrpc: Literal["2.0"] = "2.0"
    id: Optional[Union[str, int]] = None  # None ⇒ notification, no response
    method: str
    params: Optional[Dict[str, Any]] = None


class JsonRpcError(BaseModel):
    code: int
    message: str
    data: Optional[Any] = None


class JsonRpcResponse(BaseModel):
    """Outbound response or error."""
    jsonrpc: Literal["2.0"] = "2.0"
    id: Optional[Union[str, int]] = None
    result: Optional[Any] = None
    error: Optional[JsonRpcError] = None


# ── MCP common types ──────────────────────────────────────────────────

class Implementation(BaseModel):
    """Name + version of either client or server."""
    name: str
    version: str


class ServerCapabilities(BaseModel):
    """What the server supports. Each capability is an object so future
    fields can be added without breaking existing clients."""
    model_config = ConfigDict(extra="allow")

    tools:     Optional[Dict[str, Any]] = None  # {"listChanged": bool}
    resources: Optional[Dict[str, Any]] = None  # {"subscribe": bool, "listChanged": bool}
    prompts:   Optional[Dict[str, Any]] = None  # {"listChanged": bool}
    logging:   Optional[Dict[str, Any]] = None  # {}
    experimental: Optional[Dict[str, Any]] = None


class ClientCapabilities(BaseModel):
    model_config = ConfigDict(extra="allow")

    roots:    Optional[Dict[str, Any]] = None
    sampling: Optional[Dict[str, Any]] = None
    experimental: Optional[Dict[str, Any]] = None


# ── initialize ────────────────────────────────────────────────────────

class InitializeParams(BaseModel):
    protocolVersion: str
    capabilities:    ClientCapabilities = Field(default_factory=ClientCapabilities)
    clientInfo:      Implementation


class InitializeResult(BaseModel):
    protocolVersion: str
    capabilities:    ServerCapabilities
    serverInfo:      Implementation
    instructions:    Optional[str] = None


# ── tools/list, tools/call ────────────────────────────────────────────

class ToolListItem(BaseModel):
    """One tool in tools/list response. ``inputSchema`` MUST be a valid
    JSON Schema object (the spec is strict about this — clients pass it
    straight to the LLM as a function definition)."""
    name: str
    description: str = ""
    inputSchema: Dict[str, Any]  # JSON Schema


class ToolListResult(BaseModel):
    tools: List[ToolListItem]
    nextCursor: Optional[str] = None  # pagination — unused in v1


class ToolCallParams(BaseModel):
    name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)
    _meta: Optional[Dict[str, Any]] = None


class ToolContentText(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ToolContentImage(BaseModel):
    type: Literal["image"] = "image"
    data: str       # base64
    mimeType: str


class ToolCallResult(BaseModel):
    """Spec result: a list of content blocks + isError flag."""
    content: List[Union[ToolContentText, ToolContentImage]]
    isError: bool = False


# ── resources/list, resources/read ────────────────────────────────────

class Resource(BaseModel):
    """One resource in resources/list. ``uri`` is the addressing key
    used by resources/read. ``mimeType`` lets clients pick a renderer."""
    uri: str
    name: str
    description: Optional[str] = None
    mimeType: Optional[str] = None


class ResourceListResult(BaseModel):
    resources: List[Resource]
    nextCursor: Optional[str] = None


class ResourceContents(BaseModel):
    uri: str
    mimeType: Optional[str] = None
    text: Optional[str] = None      # for text resources
    blob: Optional[str] = None      # for binary resources (base64)


class ResourceReadResult(BaseModel):
    contents: List[ResourceContents]


# ── prompts/list, prompts/get ─────────────────────────────────────────

class PromptArgument(BaseModel):
    name: str
    description: Optional[str] = None
    required: bool = False


class Prompt(BaseModel):
    name: str
    description: Optional[str] = None
    arguments: List[PromptArgument] = Field(default_factory=list)


class PromptListResult(BaseModel):
    prompts: List[Prompt]
    nextCursor: Optional[str] = None


class PromptMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: ToolContentText  # we only emit text in v1


class PromptGetResult(BaseModel):
    description: Optional[str] = None
    messages: List[PromptMessage]


# ── logging/setLevel ──────────────────────────────────────────────────

class LoggingSetLevelParams(BaseModel):
    level: Literal["debug", "info", "notice", "warning", "error",
                   "critical", "alert", "emergency"]
