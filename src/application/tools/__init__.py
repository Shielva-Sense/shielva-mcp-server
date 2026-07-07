"""Tools-context use cases.

Two use cases for the MCP spec methods:
    * ListTools  — backs ``tools/list``
    * ExecuteTool — backs ``tools/call``

Both delegate to the two-port pair (ToolCatalogue + ToolExecutor) so
the application service has no idea how tools are stored or run.
"""

from .services import ToolApplicationService

__all__ = ["ToolApplicationService"]
