"""TenantContext — the isolation key threaded through every domain
operation.

Value object: equal-by-value, immutable. The frozen dataclass shape
keeps it framework-free; the interface layer's Pydantic models
construct it from request headers.

Why this is in the shared kernel:
    Every bounded context needs tenant scope. Putting TenantContext
    in any single context would force cyclic imports. The shared
    kernel exists for exactly this — types every context depends on.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

# The canonical Shielva-internal service identity. Per
# docs/architecture/MCP_LLM_BROKER.md ("Auth — internal service-to-service"),
# internal services (codex, integration, cms…) call MCP with this email. The
# gateway stamps identity from the verified JWT for EXTERNAL callers and strips
# forged X-User-Email, so a tenant cannot impersonate this — only trusted
# internal services (direct, private-network) present it.
INTERNAL_SERVICE_EMAIL = "internal@shielva.ai"


@dataclass(frozen=True, slots=True)
class TenantContext:
    tenant_id:   str
    user_id:     str
    user_email:  str
    role:        str = "Customer_Basic"
    permissions: Tuple[str, ...] = field(default_factory=tuple)

    @property
    def kb_namespace(self) -> str:
        """Per-tenant namespace for KB/vector-store partitioning."""
        return f"ns_{self.tenant_id.replace('-', '_')}"

    @property
    def is_internal_service(self) -> bool:
        """True for Shielva-internal service callers (infra, not tenant feature
        use). Internal calls — e.g. codegen RAG over connector-development
        guidelines — are NOT subject to tenant tool-permission gating, mirroring
        the /codegen/* endpoints that run with no permission gating."""
        return (self.user_email or "").strip().lower() == INTERNAL_SERVICE_EMAIL

    def has_permission(self, perm: str) -> bool:
        return perm in self.permissions
