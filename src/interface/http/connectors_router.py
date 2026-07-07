"""Connector sync + status endpoints.

Both stubs in the legacy main.py; preserved here so callers don't
404. Real implementations land when the connectors context moves
into the new layer (out of scope for this slice).
"""

from __future__ import annotations

import os

import structlog
from fastapi import APIRouter, Depends

from src.protocol.models import ConnectorSyncRequest, ConnectorSyncResponse

from ._deps import get_tenant_context

logger = structlog.get_logger(__name__)

_PREFIX = os.getenv("MCP_API_PREFIX", "/mcp/v1")

connectors_router = APIRouter(prefix=_PREFIX, tags=["connectors"])


@connectors_router.post("/connectors/sync", response_model=ConnectorSyncResponse)
async def trigger_connector_sync(
    body: ConnectorSyncRequest,
    tenant_context=Depends(get_tenant_context),
) -> ConnectorSyncResponse:
    """Stub — preserves the legacy contract while the connectors
    context is migrated."""
    logger.info(
        "Connector sync triggered",
        tenant_id=tenant_context.tenant_id,
        connector_id=body.connector_id,
    )
    return ConnectorSyncResponse(
        job_id="sync-job-123",
        status="syncing",
        documents_found=0,
        message="Sync initiated",
    )


@connectors_router.get("/connectors/{connector_id}/status")
async def get_connector_status(
    connector_id: str,
    tenant_context=Depends(get_tenant_context),
) -> dict:
    return {
        "connector_id": connector_id,
        "health": "healthy",
        "last_sync": None,
        "documents_indexed": 0,
    }
