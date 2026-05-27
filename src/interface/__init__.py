"""Interface layer — inbound adapters that translate external
protocols into application-layer use-case calls.

What lives here:
    * ``mcp_jsonrpc/``  — spec-compliant MCP JSON-RPC 2.0 façade.
    * ``http/``         — legacy REST routes (codegen, ingest, …).
                          Will move in later slices.

Interface adapters MAY import:
    * ``application/`` use cases
    * ``domain/`` types (TenantContext, value objects, error types)
    * FastAPI, Pydantic — they're the framework boundary
    * Their own DTO files (e.g. JSON-RPC envelopes)

Interface adapters MUST NOT import:
    * ``infrastructure/`` adapters directly. Get the configured
      use-case from the composition root, not from the adapter.
"""
