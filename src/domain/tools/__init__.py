"""Tools bounded context.

Owns the catalogue of tools exposed to LLM clients via MCP
``tools/list`` and ``tools/call``. A *tool* is a named, schema-typed
operation an LLM can request the server execute on its behalf.

Public surface:
    Entities      : Tool
    Value objects : ToolName, ToolSchema, ToolContentBlock,
                    ToolText, ToolImage, ToolResult
    Ports         : ToolCatalogue, ToolExecutor
    Errors        : ToolNotFoundError, ToolPermissionDeniedError

Why split into two ports (catalogue + executor) instead of one big
"ToolRegistry":
    * ``tools/list`` (read) is a different operation from
      ``tools/call`` (write/side-effecting). Different latency,
      different auth, different observability. Splitting lets a
      future adapter back the catalogue with a fast cache while the
      executor still talks to the real implementation.
    * Read-only adapters (tests, dry-runs) can implement just the
      catalogue and stub the executor.
"""

from .entities import Tool
from .errors import (
    ToolExecutionError,
    ToolNotFoundError,
    ToolPermissionDeniedError,
)
from .repositories import ToolCatalogue, ToolExecutor
from .value_objects import (
    ToolContentBlock,
    ToolImage,
    ToolName,
    ToolResult,
    ToolSchema,
    ToolText,
)

__all__ = [
    "Tool",
    "ToolCatalogue",
    "ToolContentBlock",
    "ToolExecutionError",
    "ToolExecutor",
    "ToolImage",
    "ToolName",
    "ToolNotFoundError",
    "ToolPermissionDeniedError",
    "ToolResult",
    "ToolSchema",
    "ToolText",
]
