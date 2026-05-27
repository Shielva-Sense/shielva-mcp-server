"""Composition root — the only place that knows about every layer.

This is where infrastructure adapters get wired into the application
services and bound to interface routers. No other file imports both
``infrastructure/`` adapter classes AND ``interface/`` routers — that
coupling lives only here.

Slice 1 wired the chat bounded context end-to-end (JSON-RPC router).
Slice 4b adds :func:`wire_use_cases` — the composition entry point
that ``main.py``'s lifespan calls after the legacy infra singletons
are constructed. Together the two functions own the entire new-
layer wiring graph.
"""
from .mcp_jsonrpc_wiring import build_mcp_jsonrpc_router
from .wire_use_cases     import wire_use_cases

__all__ = ["build_mcp_jsonrpc_router", "wire_use_cases"]
