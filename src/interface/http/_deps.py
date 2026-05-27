"""Shared FastAPI dependencies for the HTTP inbound adapter."""
from __future__ import annotations

import os
from typing import Any

import structlog
from fastapi import HTTPException, Request

from src.domain.shared.tenant import TenantContext as DomainTenant

logger = structlog.get_logger(__name__)


async def get_tenant_context(request: Request):
    """Extract tenant context from request headers and fetch
    permissions from MongoDB.

    Returns the *legacy* ``protocol.models.TenantContext`` (Pydantic)
    because every existing route still consumes that type. Slice 4c
    will flip routes onto the new domain dataclass and delete the
    Pydantic version. For new code that wants the domain type,
    use :func:`get_domain_tenant`.
    """
    # Lazy import — the legacy Pydantic type lives outside interface/
    from src.protocol.models import TenantContext as LegacyTenant

    tenant_id  = request.headers.get("X-Tenant-ID")
    user_id    = request.headers.get("X-User-ID")
    user_email = request.headers.get("X-User-Email")
    role       = request.headers.get("X-User-Role", "Customer_Basic")

    if not tenant_id:
        raise HTTPException(status_code=401, detail="Missing X-Tenant-ID header")

    permissions = await _fetch_permissions(user_email) if user_email else []

    return LegacyTenant(
        tenant_id   = tenant_id,
        user_id     = user_id or "unknown",
        user_email  = user_email or "unknown@example.com",
        role        = role,
        permissions = permissions,
    )


async def get_domain_tenant(request: Request) -> DomainTenant:
    """Same extraction as ``get_tenant_context``, but returns the
    domain dataclass. Use this for new routes — the legacy version
    only exists for backwards compat."""
    tenant_id  = request.headers.get("X-Tenant-ID")
    user_id    = request.headers.get("X-User-ID")
    user_email = request.headers.get("X-User-Email")
    role       = request.headers.get("X-User-Role", "Customer_Basic")

    if not tenant_id:
        raise HTTPException(status_code=401, detail="Missing X-Tenant-ID header")

    permissions = await _fetch_permissions(user_email) if user_email else []

    return DomainTenant(
        tenant_id   = tenant_id,
        user_id     = user_id or "unknown",
        user_email  = user_email or "unknown@example.com",
        role        = role,
        permissions = tuple(permissions),
    )


# ── helpers ────────────────────────────────────────────────────────

async def _fetch_permissions(user_email: str) -> list[str]:
    """Permission lookup against the legacy ``CustomerProfile.llm_user_configuration``
    Mongo collection. Best-effort: any error returns ``[]`` rather than
    failing the request (matches the legacy behaviour main.py shipped)."""
    try:
        from motor.motor_asyncio import AsyncIOMotorClient
        mongodb_url = os.getenv(
            "MONGODB_URL",
            "mongodb+srv://shielvaadmin:shielvaadmin123@mastershielva.8rbs44q.mongodb.net/?appName=MasterShielva",
        )
        try:
            import certifi as _c
            _tls = {"tlsCAFile": _c.where()} if mongodb_url.startswith("mongodb+srv") else {}
        except ImportError:
            _tls = {}
        client = AsyncIOMotorClient(mongodb_url, **_tls)
        try:
            db = client["CustomerProfile"]
            user_config = await db["llm_user_configuration"].find_one(
                {"user_email": user_email}
            )
        finally:
            client.close()
        if user_config and "access_permissions" in user_config:
            return list(user_config["access_permissions"])
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "permissions_lookup_failed",
            user_email=user_email, error=str(e),
        )
    return []
