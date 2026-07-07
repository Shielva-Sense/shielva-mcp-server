"""Knowledge entities — currently just KnowledgeBase.

Chunk is intentionally a value object (in ``value_objects.py``)
rather than an entity. We treat a chunk as a derivable artefact of
its Document — chunks don't have an identity that persists across
re-ingestions; the vector store reindexes them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..shared.tenant import TenantContext
from .value_objects import KBId, KBName, KBStatus


@dataclass(slots=True)
class KnowledgeBase:
    """A KB aggregate root.

    Owns its status transitions; mutations go through methods.
    Constructed by ``application/knowledge/provision_kb.py`` (later
    slice) and rehydrated by adapters.
    """

    id: KBId
    tenant_id: str  # not TenantContext — KBs are per-tenant
    bot_id: str
    name: KBName
    status: KBStatus
    hello_id: str  # connector-side handshake token
    unique_name: str  # connector-side resource name
    doc_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def mark_active(self) -> None:
        self.status = KBStatus.ACTIVE

    def mark_failed(self) -> None:
        self.status = KBStatus.FAILED

    def is_visible_to(self, tenant: TenantContext) -> bool:
        return tenant.tenant_id == self.tenant_id
