"""Chat-context use cases.

Slice 1 shipped session-lifecycle (ChatApplicationService).
Slice 4b adds the RAG-query orchestration use case
(:class:`HandleQueryUseCase`) — the entry point for
``POST /mcp/v1/query`` and any future MCP JSON-RPC method that
needs the full pipeline.
"""

from .handle_query import (
    HandleQueryInput,
    HandleQueryOutput,
    HandleQueryUseCase,
)
from .services import ChatApplicationService

__all__ = [
    "ChatApplicationService",
    "HandleQueryInput",
    "HandleQueryOutput",
    "HandleQueryUseCase",
]
