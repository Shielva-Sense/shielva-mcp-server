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

    def has_permission(self, perm: str) -> bool:
        return perm in self.permissions
