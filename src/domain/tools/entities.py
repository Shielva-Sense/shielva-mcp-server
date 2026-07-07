"""Tool entity — the catalogue-level record for one named tool."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..shared.tenant import TenantContext
from .value_objects import ToolName, ToolSchema


@dataclass(frozen=True, slots=True)
class Tool:
    """A tool exposed to LLM clients.

    Tools are immutable from the domain's POV — registration happens
    via adapters at startup. Dynamic registration would be a separate
    use case (and would also need ``notifications/tools/list_changed``
    which the spec defines for exactly this case).
    """

    name: ToolName
    description: str
    input_schema: ToolSchema
    required_permissions: tuple[str, ...] = field(default_factory=tuple)
    enabled_by_default: bool = True

    def is_permitted_for(self, tenant: TenantContext) -> bool:
        """A tool is permitted when the calling tenant has every
        ``required_permissions`` entry. Empty permissions = public
        to every authenticated tenant.

        Shielva-internal service callers (``tenant.is_internal_service``) bypass
        the permission gate: internal infra (e.g. the integration builder's
        codegen-guideline RAG) is not tenant feature-use, matching the
        un-gated ``/codegen/*`` internal endpoints. The identity is gateway-
        controlled, so a tenant cannot forge it."""
        if tenant.is_internal_service:
            return True
        if not self.required_permissions:
            return True
        return all(tenant.has_permission(p) for p in self.required_permissions)
