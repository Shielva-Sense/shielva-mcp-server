"""KB + Bot provisioning + test + activate routes.

These thin wrappers delegate to the legacy ``MessageHandler``'s
provisioning helpers (``handle_provision_kb`` / ``handle_provision_bot``
/ ``handle_test_bot``). Slice 4c+ will lift provisioning into proper
``application/knowledge`` + ``application/bots`` use cases; until
then the route stays consumer-compatible.
"""

from __future__ import annotations

import os

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request

from src.protocol.models import (
    ProvisionBotRequest,
    ProvisionBotResponse,
    ProvisionKBRequest,
    ProvisionKBResponse,
    TestBotRequest,
    TestBotResponse,
)

from ._deps import get_tenant_context

logger = structlog.get_logger(__name__)

_PREFIX = os.getenv("MCP_API_PREFIX", "/mcp/v1")

provision_router = APIRouter(prefix=_PREFIX, tags=["provisioning"])


def _handler(request: Request):
    h = getattr(request.app.state, "message_handler", None)
    if h is None:
        raise HTTPException(status_code=503, detail="MessageHandler not initialised")
    return h


@provision_router.post("/provision/kb", response_model=ProvisionKBResponse)
async def provision_kb(
    request: Request,
    body: ProvisionKBRequest,
    tenant_context=Depends(get_tenant_context),
) -> ProvisionKBResponse:
    logger.info(
        "KB provision request",
        tenant_id=tenant_context.tenant_id,
        kb_name=body.kb_config.name,
    )
    try:
        return await _handler(request).handle_provision_kb(
            request=body,
            tenant_context=tenant_context,
        )
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        logger.error("KB provisioning failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@provision_router.post("/provision/bot", response_model=ProvisionBotResponse)
async def provision_bot(
    request: Request,
    body: ProvisionBotRequest,
    tenant_context=Depends(get_tenant_context),
) -> ProvisionBotResponse:
    logger.info(
        "Bot provision request",
        tenant_id=tenant_context.tenant_id,
        bot_name=body.name,
    )
    try:
        return await _handler(request).handle_provision_bot(
            request=body,
            tenant_context=tenant_context,
        )
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@provision_router.post("/test", response_model=TestBotResponse)
async def test_bot(
    request: Request,
    body: TestBotRequest,
    tenant_context=Depends(get_tenant_context),
) -> TestBotResponse:
    logger.info(
        "Bot test request",
        tenant_id=tenant_context.tenant_id,
        bot_id=body.bot_id,
    )
    try:
        return await _handler(request).handle_test_bot(
            request=body,
            tenant_context=tenant_context,
        )
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@provision_router.post("/activate")
async def activate_bot(
    bot_id: str,
    tenant_context=Depends(get_tenant_context),
) -> dict:
    """Activate a bot after successful testing. The legacy handler
    has a TODO here; preserving that until the bots context grows
    a real activation use case."""
    logger.info(
        "Bot activation request",
        tenant_id=tenant_context.tenant_id,
        bot_id=bot_id,
    )
    return {"status": "activated", "bot_id": bot_id}
