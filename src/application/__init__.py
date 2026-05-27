"""Application layer — use cases that orchestrate the domain.

Each module corresponds to a bounded context. Use cases:
    * Take primitive inputs (or DTOs from the interface adapter)
    * Coordinate the domain aggregates + ports
    * Return primitive outputs (or domain value objects)
    * NEVER touch HTTP, JSON-RPC, FastAPI, Pydantic.BaseModel
      response shapes, structlog binding, etc. — those belong in
      adapters.

The application layer is the only place that knows BOTH the domain
ports AND how a workflow stitches them together. The interface
layer calls into application; application calls into domain +
ports; ports are implemented by infrastructure.
"""
