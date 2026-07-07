"""Shared FastAPI dependencies for the HTTP inbound adapter.

The two ``get_tenant_context`` / ``get_domain_tenant`` legacy helpers are
kept for backwards compatibility, BUT they now go through
``shielva_common.auth.require_principal`` which:

* Verifies the gateway-stamped HMAC signature (when configured).
* Rejects requests without canonical identity headers.
* Builds a frozen ``Principal`` that downstream code cannot mutate.

The MongoDB URI is loaded via SealedSettings — no plaintext fallback. If
the env was not unsealed before import time, settings load fails LOUD.

Permissions look-ups are delegated to
``MongoUserConfigRepository`` via ``get_user_config_repo()`` — no
inline DB calls here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog
from fastapi import Depends, Request
from shielva_common.auth import Principal, require_principal

from src.domain.shared.tenant import TenantContext as DomainTenant
from src.infrastructure.persistence.mongo_user_config_repository import (
    MongoUserConfigRepository,
    get_user_config_repo,
)

if TYPE_CHECKING:
    from src.protocol.models import TenantContext as LegacyTenant

logger = structlog.get_logger(__name__)


async def get_tenant_context(
    request: Request,
    user_config_repo: MongoUserConfigRepository = Depends(get_user_config_repo),
) -> LegacyTenant:
    """Extract tenant context from gateway-verified headers and fetch
    permissions from MongoDB.

    Returns the legacy ``protocol.models.TenantContext`` (Pydantic) for
    existing route consumers. New code should use ``get_domain_tenant``.
    """
    from src.protocol.models import TenantContext as LegacyTenant

    principal: Principal = require_principal(request)
    permissions = await user_config_repo.get_permissions(principal.email)

    role = request.headers.get("X-Shielva-Roles") or request.headers.get("X-User-Role", "Customer_Basic")

    return LegacyTenant(
        tenant_id=principal.tenant_id,
        user_id=principal.user_id,
        user_email=principal.email,
        role=role,
        permissions=permissions,
    )


async def get_domain_tenant(
    request: Request,
    user_config_repo: MongoUserConfigRepository = Depends(get_user_config_repo),
) -> DomainTenant:
    """Domain-typed tenant context. Use this for new routes."""
    principal: Principal = require_principal(request)
    permissions = await user_config_repo.get_permissions(principal.email)

    role = request.headers.get("X-Shielva-Roles") or request.headers.get("X-User-Role", "Customer_Basic")

    return DomainTenant(
        tenant_id=principal.tenant_id,
        user_id=principal.user_id,
        user_email=principal.email,
        role=role,
        permissions=tuple(permissions),
    )
