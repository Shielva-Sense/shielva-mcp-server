"""Spec-compliant MCP JSON-RPC 2.0 inbound adapter.

Implements the Model Context Protocol (spec 2024-11-05 with
Streamable HTTP 2025-03-26) as a FastAPI router. The handlers are
thin: they validate the JSON-RPC envelope, translate to domain
types, call the application layer, and serialise the result.

Spec methods implemented:
    initialize, notifications/initialized, ping, tools/list,
    tools/call, resources/list, resources/read, prompts/list,
    prompts/get, logging/setLevel, notifications/cancelled.
"""
from .transport import build_router

__all__ = ["build_router"]
