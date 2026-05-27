"""Inbound HTTP/REST adapters.

Every REST endpoint shielva-mcp serves lives under this package
(except the MCP JSON-RPC ``POST /mcp`` façade which has its own
package). Each module owns one logical surface; main.py composes
them in via ``app.include_router``.

Why these stay alongside the MCP JSON-RPC layer rather than being
folded into it:
    * They aren't MCP-spec methods. They're internal control-plane
      surfaces with their own consumer set (integration-builder,
      ingestion-worker, presence-core, shielva-cms).
    * Where they overlap with MCP-spec methods (``GET /tools``,
      ``POST /tools/{name}/execute``) the implementations call the
      same application-layer use cases (ToolApplicationService) so
      behaviour is consistent across surfaces.
"""
from .admin_router      import admin_router
from .codegen_router    import codegen_router
from .connectors_router import connectors_router
from .embeddings_router import embeddings_router
from .health_router     import health_router
from .ingest_router     import router as ingest_router, start_scheduler, stop_scheduler
from .llm_router        import llm_router
from .provision_router  import provision_router
from .query_router      import query_router
from .tools_router      import tools_router

__all__ = [
    "admin_router",
    "codegen_router",
    "connectors_router",
    "embeddings_router",
    "health_router",
    "ingest_router", "start_scheduler", "stop_scheduler",
    "llm_router",
    "provision_router",
    "query_router",
    "tools_router",
]
