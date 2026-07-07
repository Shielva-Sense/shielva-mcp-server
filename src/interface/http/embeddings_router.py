"""Embeddings batch endpoint.

Used by sibling services (shielva-presence-core TrailFollower,
InlineQARouter) that need vectors in the same space as MCP's
retrieval layer. Single-method router; the use case is read-only +
side-effect-free so it bypasses the policy gate.
"""

from __future__ import annotations

import os

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from ._deps import get_tenant_context

logger = structlog.get_logger(__name__)


_PREFIX = os.getenv("MCP_API_PREFIX", "/mcp/v1")

embeddings_router = APIRouter(prefix=_PREFIX, tags=["embeddings"])


# Wire shape — matches the existing client contract (presence-core
# already calls this endpoint with this body shape).
class EmbeddingsRequest(BaseModel):
    texts: list[str] = Field(
        ...,
        description="Texts to embed. Capped at 256 per request.",
        min_length=1,
    )


class EmbeddingsResponse(BaseModel):
    embeddings: list[list[float]]
    dimension: int
    model: str
    count: int


# Cap per request — keeps latency bounded + prevents OOM via a
# pathological caller. The underlying EmbeddingClient batches
# internally (config.batch_size = 100), so this is a safety net.
_MAX_PER_REQUEST = 256


@embeddings_router.post("/embeddings", response_model=EmbeddingsResponse)
async def generate_embeddings(
    body: EmbeddingsRequest,
    request: Request,
    tenant_context=Depends(get_tenant_context),
) -> EmbeddingsResponse:
    """Embed ``body.texts`` against the configured provider. Auth via
    X-Tenant-ID; no role gate — embeddings are a primitive, not a
    sensitive op."""
    client = getattr(request.app.state, "embedding_client", None)
    if client is None:
        logger.error(
            "embedding_client_not_initialised",
            tenant=tenant_context.tenant_id,
        )
        raise HTTPException(
            status_code=503,
            detail="Embedding client not initialised — MCP server failed to start cleanly.",
        )

    if not body.texts:
        return EmbeddingsResponse(
            embeddings=[],
            dimension=client.config.dimension,
            model=client.config.model,
            count=0,
        )

    if len(body.texts) > _MAX_PER_REQUEST:
        raise HTTPException(
            status_code=413,
            detail=f"Too many texts in one request (got {len(body.texts)}, max {_MAX_PER_REQUEST}).",
        )

    try:
        vectors = await client.embed(body.texts)
    except Exception as e:
        logger.error(
            "embedding_generation_failed",
            error=str(e),
            tenant=tenant_context.tenant_id,
        )
        raise HTTPException(status_code=500, detail=f"Embedding generation failed: {e}")

    return EmbeddingsResponse(
        embeddings=vectors,
        dimension=client.config.dimension,
        model=client.config.model,
        count=len(vectors),
    )
